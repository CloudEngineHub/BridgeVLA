'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/mvt_single.py
Therefore, the code is also under the NVIDIA Source Code License


Architecture notes:
- Rotation / gripper / collision predictions use the original
  ``feat_fc`` family: select_feat_from_hm + feat_fc +
  feat_fc_x/y/z + feat_fc_ex_rot.
- ``use_modified_focal_loss`` (ablation, default False) switches the
  heatmap head between the original single ``up0`` and the dual
  ``up_action`` / ``up_grounding`` pair (built from heads_focal.py).
- ``use_view_logvar`` (ablation, default False) is a placeholder;
  when True the constructor raises NotImplementedError.
'''
import os
import json
import contextlib
import torch
from torch import nn
from einops import rearrange

import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.attn import FixedPositionalEncoding
from bridgevla.mvt.raft_utils import ConvexUpSample
from bridgevla.mvt.heads_focal import build_focal_dual_head
from bridgevla.mvt.view_logvar import assert_view_logvar_disabled
from bridgevla.mvt.memory import MemoryBlock
from bridgevla.mvt.keyframe_disc import KeyframeDiscriminator
from PIL import Image


class _LMHeadStub(nn.Module):
    """Drop-in replacement for PaliGemma's ``lm_head`` used ONLY while we run
    the trunk forward purely to harvest ``hidden_states``. The real lm_head is
    a (K=2048 -> N=vocab=257216) projection; at eval batch sizes that skinny
    GEMM deterministically SIGFPEs on this cluster's H20-3e (see the call site
    in ``_paligemma_extract``). The produced logits are never read, so we
    return a cheap shape-preserving stand-in: (*input.shape[:-1], 1). HF only
    forwards this into ``CausalLMOutputWithPast.logits`` (unused here), so the
    singleton vocab dim is harmless and keeps us off the broken kernel."""

    def forward(self, x):
        return x.new_zeros((*x.shape[:-1], 1))


class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        img_feat_dim,
        feat_dim,
        im_channels,
        activation,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        rend_oblique_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        num_arms=1,
        predict_collision=True,
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
        load_pretrain=False,
        pretrain_path=None,
        use_modified_focal_loss=False,
        use_view_logvar=False,
        memory_cfg=None,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.feat_ver = feat_ver
        self.wpt_img_aug = wpt_img_aug
        self.inp_pre_pro = inp_pre_pro
        self.inp_pre_con = inp_pre_con
        self.cvx_up = cvx_up
        self.use_point_renderer = use_point_renderer
        self.rot_ver = rot_ver
        self.num_rot = num_rot
        # ---- Dual-arm gating ----
        # num_arms == 1 is the original single-arm BridgeVLA (flat head names, byte-identical). num_arms > 1
        # (RMBench) adds per-arm up0 heads at both stages and per-arm rot/grip heads at stage 2; the
        # PaliGemma trunk + memory blocks stay shared (orchestrated in bridgevla.mvt.mvt.MVT).
        self.num_arms = int(num_arms)
        # Introspection only — nothing here reads it, and it does not change the feat head's output width.
        # Collision supervision is gated agent-side by ``RVTAgent(predict_collision=...)``.
        self.predict_collision = bool(predict_collision)
        self.no_feat = no_feat
        self.use_modified_focal_loss = use_modified_focal_loss
        self.use_view_logvar = use_view_logvar
        # `rend_oblique_views` is consumed by mvt.MVT when it builds the renderer; kept here for signature parity.
        self.rend_oblique_views = rend_oblique_views

        # Kendall-Gal adaptive weighting is deferred — fail fast before allocating any heads.
        assert_view_logvar_disabled(self.use_view_logvar)

        if self.cvx_up:
            assert not self.inp_pre_con, (
                "When using the convex upsampling, we do not concatenate"
                " features from input_preprocess to the features used for"
                " prediction"
            )

        _rank = (torch.distributed.get_rank()
                 if (torch.distributed.is_available()
                     and torch.distributed.is_initialized())
                 else 0)
        if _rank == 0:
            print(f"MVT Vars: {vars(self)}")

        assert renderer is not None
        self.renderer = renderer
        self.num_img = self.renderer.num_img
        # 16**2 patches per image (PaliGemma-3B-pt-224 ViT).
        self.num_pat_img = 16

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # PaliGemma image-token hidden size.
        self.vlm_dim = 2048

        # ---- Heatmap head(s) ----
        # Default (use_modified_focal_loss=False): a single up0 head as in the original BridgeVLA, softmaxed
        # over (h*w) and supervised by soft-label CE. True: dual up_action / up_grounding heads from
        # heads_focal.py with the HM_PRIOR_LOGIT bias prior, supervised by sigmoid + modified focal loss.
        if self.use_modified_focal_loss:
            # The focal dual-head ablation is single-arm only (real-finetune).
            assert self.num_arms == 1, (
                "use_modified_focal_loss is not supported with num_arms > 1"
            )
            self.up_action, self.up_grounding = build_focal_dual_head(
                in_dim=self.vlm_dim, up_ratio=self.img_patch_size,
            )
        elif self.num_arms == 1:
            # Single-arm: flat ``up0`` name keeps old checkpoints loadable.
            self.up0 = ConvexUpSample(
                in_dim=self.vlm_dim,
                out_dim=1,
                up_ratio=self.img_patch_size,
            )
        else:
            # Dual-arm: one non-shared heatmap head per arm, used at BOTH stages (stage-1 trunk tokens are shared).
            self.up0_arms = nn.ModuleList([
                ConvexUpSample(
                    in_dim=self.vlm_dim,
                    out_dim=1,
                    up_ratio=self.img_patch_size,
                )
                for _ in range(self.num_arms)
            ])

        # ---- Rotation / gripper / collision heads (original BridgeVLA). ----
        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.vlm_dim
            # Concat max-pooled image tokens with the image tokens at the waypoint pixel.
            if self.cvx_up:
                feat_fc_dim += self.vlm_dim
            else:
                feat_fc_dim += self.final_dim

            def get_feat_fc(_feat_in_size, _feat_out_size,
                             _feat_fc_dim=feat_fc_dim):
                """Three-layer MLP factory matching the original BridgeVLA."""
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size),
                ]
                return nn.Sequential(*layers)

            feat_out_size = feat_dim
            # ``predict_collision`` does NOT change the head width: the feat head always emits
            # grip(2)+collision(2). Benches that set it False leave those logits unsupervised and unused,
            # which keeps feat_dim / get_num_feat untouched and checkpoints loadable across benches.
            if self.rot_ver in (0, 2):
                # rot_ver == 0: feat_out_size = num_rot*3 + grip(2) + coll(2). rot_ver == 2 (6D regression):
                # 6 + 2 + 2 = 10, set via mvt_cfg.feat_dim. Same feat_fc MLP, only the output width differs.
                if self.num_arms == 1:
                    self.feat_fc = get_feat_fc(
                        self.num_img * feat_fc_dim,
                        feat_out_size,
                    )
                else:
                    self.feat_fc_arms = nn.ModuleList([
                        get_feat_fc(self.num_img * feat_fc_dim, feat_out_size)
                        for _ in range(self.num_arms)
                    ])
            elif self.rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)

                # feat_fc_pe is parameter-free (only a div_term buffer), so it is shared across arms.
                self.feat_fc_pe = FixedPositionalEncoding(
                    self.num_img * feat_fc_dim, feat_scale_factor=1
                )

                def _build_rot_heads():
                    """One arm's rot/grip(/coll) head set (rot_ver==1)."""
                    ex_rot = (
                        get_feat_fc(self.num_img * feat_fc_dim, feat_out_size_ex_rot)
                        if feat_out_size_ex_rot > 0 else None
                    )
                    return nn.ModuleDict({
                        **({"ex_rot": ex_rot} if ex_rot is not None else {}),
                        "init_bn": nn.BatchNorm1d(self.num_img * feat_fc_dim),
                        "x": get_feat_fc(self.num_img * feat_fc_dim, self.num_rot),
                        "y": get_feat_fc(self.num_img * feat_fc_dim, self.num_rot),
                        "z": get_feat_fc(self.num_img * feat_fc_dim, self.num_rot),
                    })

                if self.num_arms == 1:
                    # Flat attribute names — keeps old checkpoints loadable.
                    if feat_out_size_ex_rot > 0:
                        self.feat_fc_ex_rot = get_feat_fc(
                            self.num_img * feat_fc_dim, feat_out_size_ex_rot
                        )
                    self.feat_fc_init_bn = nn.BatchNorm1d(
                        self.num_img * feat_fc_dim
                    )
                    self.feat_fc_x = get_feat_fc(
                        self.num_img * feat_fc_dim, self.num_rot
                    )
                    self.feat_fc_y = get_feat_fc(
                        self.num_img * feat_fc_dim, self.num_rot
                    )
                    self.feat_fc_z = get_feat_fc(
                        self.num_img * feat_fc_dim, self.num_rot
                    )
                else:
                    # Per-arm rot/grip heads (non-shared).
                    self.feat_fc_rot_arms = nn.ModuleList(
                        [_build_rot_heads() for _ in range(self.num_arms)]
                    )
            else:
                assert False

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from bridgevla.mvt.renderer import select_feat_from_hm

        from transformers import (
            PaliGemmaProcessor,
            PaliGemmaForConditionalGeneration,
        )
        from safetensors import safe_open

        def load_safetensors_shards(checkpoint_dir):
            """Load HF-Trainer ``model.safetensors.index.json`` layout."""
            with open(f"{checkpoint_dir}/model.safetensors.index.json") as f:
                index = json.load(f)
            all_params = {}
            for shard_file in set(index["weight_map"].values()):
                with safe_open(f"{checkpoint_dir}/{shard_file}",
                               framework="pt") as f:
                    for key in f.keys():
                        clean_key = key.replace("module.", "")
                        all_params[clean_key] = f.get_tensor(key)
            return all_params

        def load_pretrain_state_dict(path):
            """Load a pretrain checkpoint from .pth, .safetensors, or HF dir.

            Accepts both this repo's pretrain layout and the original
            BridgeVLA release layout — see
            ``mvt_utils.normalize_pretrain_state_dict``.
            """
            if os.path.isdir(path):
                sd = load_safetensors_shards(path)
            elif path.endswith(".safetensors"):
                sd = {}
                with safe_open(path, framework="pt") as f:
                    for k in f.keys():
                        sd[k.replace("module.", "")] = f.get_tensor(k)
            else:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    ckpt = ckpt["model_state"]
                sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
            sd, n_rewritten = mvt_utils.normalize_pretrain_state_dict(sd)
            if n_rewritten and _rank == 0:
                print(f"[mvt_single] Original-BridgeVLA key layout detected: "
                      f"re-prefixed {n_rewritten} backbone keys with `model.`.")
            return sd

        # Local PaliGemma snapshot (avoid HF hub).
        model_id = os.environ.get("PALIGEMMA_PATH", "google/paligemma-3b-pt-224")
        if _rank == 0:
            print(f"[mvt_single] Loading PaliGemma from: {model_id}")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)

        # PaliGemma activation checkpointing. ``use_reentrant=False`` is mandatory here: DDP
        # find_unused_parameters=True is incompatible with the reentrant autograd path, and the fully-frozen
        # Stage-1 schedule drives PaliGemma with no input requiring grad (reentrant raises, non-reentrant
        # degrades to a plain forward). ``use_cache`` must be off. Hooking ``_paligemma_extract`` rather than
        # the constructor also covers the anchor / history forwards through the same ``self.model``.
        self.gradient_checkpointing_enabled = bool(gradient_checkpointing)
        if self.gradient_checkpointing_enabled:
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            if _rank == 0:
                print("[mvt_single] PaliGemma gradient checkpointing: "
                      "enabled (use_reentrant=False, use_cache=False)")
        else:
            if _rank == 0:
                print("[mvt_single] PaliGemma gradient checkpointing: disabled")

        # ---- Episodic memory blocks (spatial anchor + temporal episodic) ----
        # See ``bridgevla.mvt.memory`` for design notes. Default: stage 1 has both blocks, stage 2 spatial only
        # with independent params. Residual exits use their submodules' default init, so the memory branch
        # perturbs the pretrained path from step 0. Built BEFORE load_pretrain so its buckets can populate them.
        self.memory_cfg = memory_cfg or {}
        self.memory_enabled = bool(self.memory_cfg.get("enabled", False))
        # True: anchor / history PaliGemma forwards build a graph and the KV tokens are not detached, so
        # gradient flows back into PaliGemma. False: legacy no_grad + detach. Default True so MemoryBench
        # trains end-to-end through the memory KV path (trade-off note in bridgevla/mvt/config.py).
        self.memory_grad_through_tokens = bool(
            self.memory_cfg.get("grad_through_tokens", True)
        )
        if self.memory_enabled:
            mem_heads = int(self.memory_cfg.get("heads", 8))
            mem_dim_head = int(self.memory_cfg.get("dim_head", 128))
            mem_num_layers = int(self.memory_cfg.get("num_layers", 2))
            mem_ffn_mult = int(self.memory_cfg.get("ffn_mult", 2))
            mem_use_fast = bool(self.memory_cfg.get("use_fast_attn", False))
            k_temporal = int(self.memory_cfg.get("k_temporal", 2))
            share_spatial = bool(
                self.memory_cfg.get("share_spatial_across_stages", False)
            )
            mem_kw = dict(
                dim=self.vlm_dim,
                heads=mem_heads,
                dim_head=mem_dim_head,
                num_layers=mem_num_layers,
                ffn_mult=mem_ffn_mult,
                use_fast=mem_use_fast,
            )
            # --- Memory ablation master switches (per-stage groups) ---
            # effective_flag = fine_grained_flag AND master_switch. Disabling a group skips constructing that
            # MemoryBlock; _apply_memory gates every injection on the block's presence, so a missing block is
            # a clean no-op at train and eval and is absent from the checkpoint. The zoom pipeline is untouched.
            #   temporal_memory -> stage-1 memory (mem_temporal_s1 + mem_spatial_s1)
            #   spatial_memory  -> stage-2 per-view spatial anchor (mem_spatial_s2)
            temporal_mem_on = bool(self.memory_cfg.get("temporal_memory", True))
            spatial_mem_on = bool(self.memory_cfg.get("spatial_memory", True))
            if self.memory_cfg.get("spatial_at_mvt1", True) and temporal_mem_on:
                self.mem_spatial_s1 = MemoryBlock(kind="spatial", **mem_kw)
            if self.memory_cfg.get("spatial_at_mvt2", True) and spatial_mem_on:
                if share_spatial and hasattr(self, "mem_spatial_s1"):
                    self.mem_spatial_s2 = self.mem_spatial_s1
                else:
                    self.mem_spatial_s2 = MemoryBlock(kind="spatial", **mem_kw)
            if self.memory_cfg.get("temporal_at_mvt1", True) and temporal_mem_on:
                self.mem_temporal_s1 = MemoryBlock(
                    kind="temporal", K_temporal=k_temporal,
                    **mem_kw,
                )

            # Mirror PaliGemma's gradient_checkpointing onto the memory blocks (loss-less recompute); one flag
            # controls both backbones.
            for _mb_attr in ("mem_spatial_s1", "mem_spatial_s2",
                             "mem_temporal_s1"):
                _mb = getattr(self, _mb_attr, None)
                if _mb is not None:
                    _mb.grad_checkpoint = self.gradient_checkpointing_enabled

            # ---- Keyframe discriminator ("is the current frame a subtask boundary worth admitting?") ----
            # Reads the POST-memory stage-1 token grid (x_heads), trained with BCE on mem_label, gates
            # MemoryBank.push at eval, lives in fp32. COUPLED to temporal_memory: with the stage-1 temporal
            # memory ablated away it has nothing to gate, so it is forced off and its BCE term leaves the loss.
            disc_cfg = dict(self.memory_cfg.get("discriminator", {}) or {})
            self.discriminator_enabled = bool(
                disc_cfg.get("enabled", False)
            ) and temporal_mem_on
            if (bool(disc_cfg.get("enabled", False)) and not temporal_mem_on
                    and _rank == 0):
                print("[mvt_single] Keyframe discriminator force-DISABLED "
                      "because memory.temporal_memory=False (nothing to gate).")
            if self.discriminator_enabled:
                self.keyframe_disc = KeyframeDiscriminator(
                    dim=self.vlm_dim,
                    num_views=self.num_img,
                    heads=int(disc_cfg.get("heads", mem_heads)),
                    hidden=int(disc_cfg.get("hidden", 512)),
                )
        else:
            self.discriminator_enabled = False

        if load_pretrain:
            assert pretrain_path is not None
            if _rank == 0:
                print(f"[mvt_single] Loading pretrain weights from: {pretrain_path}")
            all_params = load_pretrain_state_dict(pretrain_path)

            # Bucket parameters by submodule prefix; keys from removed submodules fall through to `extras`
            # and are warned about. When the ckpt only has up0.* but the model uses up_action/up_grounding,
            # both focal heads warm-start from up0.* — and the reverse direction is supported too.
            buckets = {
                "model.": {},
                "up0.": {},
                "up_action.": {},
                "up_grounding.": {},
                "feat_fc.": {},
                "feat_fc_ex_rot.": {},
                "feat_fc_init_bn.": {},
                "feat_fc_pe.": {},
                "feat_fc_x.": {},
                "feat_fc_y.": {},
                "feat_fc_z.": {},
                # Memory blocks (pretrain produces mem_spatial_s1 + mem_temporal_s1; mem_spatial_s2 is
                # cross-loaded from s1 below when the ckpt has no s2 bucket).
                "mem_spatial_s1.": {},
                "mem_spatial_s2.": {},
                "mem_temporal_s1.": {},
            }
            extras = {}
            for k, v in all_params.items():
                matched = False
                for prefix, bucket in buckets.items():
                    if k.startswith(prefix):
                        bucket[k[len(prefix):]] = v
                        matched = True
                        break
                if matched:
                    continue
                extras[k] = v

            def _load(submodule, params, name):
                if not params:
                    if _rank == 0:
                        print(f"[mvt_single] No pretrain weights for {name}; skipping.")
                    return
                # Filter shape-mismatched params: ``strict=False`` only ignores missing/unexpected keys, not
                # shape mismatches. Legacy heads with a different input dim silently re-init from scratch.
                target_sd = submodule.state_dict()
                kept, dropped_shape = {}, []
                for k, v in params.items():
                    if k in target_sd and target_sd[k].shape != v.shape:
                        dropped_shape.append(
                            (k, tuple(v.shape), tuple(target_sd[k].shape))
                        )
                        continue
                    kept[k] = v
                miss, unexp = submodule.load_state_dict(kept, strict=False)
                if _rank == 0:
                    print(f"[mvt_single] Loaded {name}: missing={len(miss)}, "
                          f"unexpected={len(unexp)}, "
                          f"shape-dropped={len(dropped_shape)}")
                    if dropped_shape:
                        for k, ck_sh, cur_sh in dropped_shape[:3]:
                            print(f"    shape mismatch -> reset: {k} "
                                  f"ckpt={ck_sh} cur={cur_sh}")

            def _prefix_bucket(prefix):
                plen = len(prefix)
                return {k[plen:]: v for k, v in all_params.items()
                        if k.startswith(prefix)}

            def _cross_load_module_list(arms, src_params, src_name, dst_name):
                """Warm-start every entry in ``arms`` from a single-arm bucket."""
                if not src_params:
                    return
                for i, head in enumerate(arms):
                    _load(head, src_params,
                          f"{dst_name}[{i}] <- {src_name} (cross-load)")

            # A warm start that quietly finds no backbone weights is the worst outcome — training would
            # proceed from base PaliGemma and look healthy. Refuse instead.
            if not buckets["model."]:
                found = sorted({k.split(".")[0] for k in all_params})
                raise RuntimeError(
                    "load_pretrain=True but the checkpoint contains no "
                    f"PaliGemma weights: {pretrain_path}\n"
                    f"  top-level key prefixes found: {found}\n"
                    "  expected either this repo's pretrain layout (`model.*`) "
                    "or the original BridgeVLA release layout "
                    "(`vision_tower.* / language_model.* / "
                    "multi_modal_projector.*`)."
                )

            _load(self.model, buckets["model."], "PaliGemma (model)")

            # Heatmap head: pick the bucket matching the current head type, with cross-loading fallback.
            if self.use_modified_focal_loss:
                if buckets["up_action."] or buckets["up_grounding."]:
                    _load(self.up_action, buckets["up_action."],
                          "up_action <- up_action")
                    _load(self.up_grounding, buckets["up_grounding."],
                          "up_grounding <- up_grounding")
                elif buckets["up0."]:
                    _load(self.up_action, buckets["up0."],
                          "up_action <- up0 (cross-load)")
                    _load(self.up_grounding, buckets["up0."],
                          "up_grounding <- up0 (cross-load)")
            elif self.num_arms == 1:
                if buckets["up0."]:
                    _load(self.up0, buckets["up0."], "up0 <- up0")
                elif buckets["up_action."]:
                    _load(self.up0, buckets["up_action."],
                          "up0 <- up_action (cross-load)")
            else:
                up0_arms_bucket = _prefix_bucket("up0_arms.")
                if up0_arms_bucket:
                    _load(self.up0_arms, up0_arms_bucket,
                          "up0_arms <- up0_arms")
                elif buckets["up0."]:
                    _cross_load_module_list(
                        self.up0_arms, buckets["up0."], "up0", "up0_arms")
                elif buckets["up_action."]:
                    _cross_load_module_list(
                        self.up0_arms, buckets["up_action."],
                        "up_action", "up0_arms")

            # Original BridgeVLA feat_fc heads.
            if not self.no_feat and self.rot_ver == 1:
                if self.num_arms == 1:
                    _load(self.feat_fc_init_bn, buckets["feat_fc_init_bn."],
                          "feat_fc_init_bn")
                    _load(self.feat_fc_x, buckets["feat_fc_x."], "feat_fc_x")
                    _load(self.feat_fc_y, buckets["feat_fc_y."], "feat_fc_y")
                    _load(self.feat_fc_z, buckets["feat_fc_z."], "feat_fc_z")
                    if (hasattr(self, "feat_fc_ex_rot")
                            and buckets["feat_fc_ex_rot."]):
                        _load(self.feat_fc_ex_rot, buckets["feat_fc_ex_rot."],
                              "feat_fc_ex_rot")
                else:
                    rot_arms_bucket = _prefix_bucket("feat_fc_rot_arms.")
                    if rot_arms_bucket:
                        _load(self.feat_fc_rot_arms, rot_arms_bucket,
                              "feat_fc_rot_arms <- feat_fc_rot_arms")
                    else:
                        for i, arm_heads in enumerate(self.feat_fc_rot_arms):
                            tag = f"feat_fc_rot_arms[{i}]"
                            _load(arm_heads["init_bn"],
                                  buckets["feat_fc_init_bn."],
                                  f"{tag}.init_bn <- feat_fc_init_bn")
                            _load(arm_heads["x"], buckets["feat_fc_x."],
                                  f"{tag}.x <- feat_fc_x")
                            _load(arm_heads["y"], buckets["feat_fc_y."],
                                  f"{tag}.y <- feat_fc_y")
                            _load(arm_heads["z"], buckets["feat_fc_z."],
                                  f"{tag}.z <- feat_fc_z")
                            if ("ex_rot" in arm_heads
                                    and buckets["feat_fc_ex_rot."]):
                                _load(arm_heads["ex_rot"],
                                      buckets["feat_fc_ex_rot."],
                                      f"{tag}.ex_rot <- feat_fc_ex_rot")
            elif not self.no_feat and self.rot_ver in (0, 2):
                if self.num_arms == 1 and buckets["feat_fc."]:
                    _load(self.feat_fc, buckets["feat_fc."], "feat_fc")
                elif self.num_arms > 1:
                    feat_fc_arms_bucket = _prefix_bucket("feat_fc_arms.")
                    if feat_fc_arms_bucket:
                        _load(self.feat_fc_arms, feat_fc_arms_bucket,
                              "feat_fc_arms <- feat_fc_arms")
                    elif buckets["feat_fc."]:
                        _cross_load_module_list(
                            self.feat_fc_arms, buckets["feat_fc."],
                            "feat_fc", "feat_fc_arms")

            # Episodic-memory blocks. Pretrain produces mem_spatial_s1 + mem_temporal_s1 only; finetune
            # cross-loads mem_spatial_s2 from mem_spatial_s1. Legacy ``action_proj.*`` buckets and mismatched
            # ``slot_embed`` shapes (different K) fall through _load's shape-drop path and re-init.
            if self.memory_enabled:
                if hasattr(self, "mem_spatial_s1") and buckets["mem_spatial_s1."]:
                    _load(self.mem_spatial_s1, buckets["mem_spatial_s1."],
                          "mem_spatial_s1 <- mem_spatial_s1")
                if hasattr(self, "mem_temporal_s1") and buckets["mem_temporal_s1."]:
                    _load(self.mem_temporal_s1, buckets["mem_temporal_s1."],
                          "mem_temporal_s1 <- mem_temporal_s1")
                if hasattr(self, "mem_spatial_s2"):
                    if buckets["mem_spatial_s2."]:
                        _load(self.mem_spatial_s2, buckets["mem_spatial_s2."],
                              "mem_spatial_s2 <- mem_spatial_s2")
                    elif buckets["mem_spatial_s1."]:
                        _load(self.mem_spatial_s2, buckets["mem_spatial_s1."],
                              "mem_spatial_s2 <- mem_spatial_s1 (cross-load)")

            if _rank == 0:
                if extras:
                    print(f"[mvt_single] Ignored {len(extras)} unrecognized keys "
                          f"(e.g. {list(extras.keys())[:3]}).")
        else:
            if _rank == 0:
                print("You are loading original paligemma model "
                      "(no pretrain weights).")

        # Episodic memory was constructed earlier (right after PaliGemma loading) so load_pretrain can
        # populate mem_spatial_s1 / mem_temporal_s1 directly. Don't reconstruct here.

        global select_feat_from_hm

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """Project 3D points to (per-view) image-plane coords."""
        return self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )

    @staticmethod
    def trans_cuda_tensor_2_PIL(cuda_tensor):
        tensor_cpu = cuda_tensor.cpu()
        image = tensor_cpu.permute(1, 2, 0).numpy()
        image = (image * 255).astype('uint8')
        pil_image = Image.fromarray(image)
        return pil_image.convert("RGB")

    def _heatmap_head(self, x, head, arm=0):
        """Run the heatmap head selected by config + per-call ``head`` kwarg.

        ``head`` is only consulted when use_modified_focal_loss=True and
        chooses between up_action (default) and up_grounding (used by the
        real-finetune Objects365 fv mix). When use_modified_focal_loss=False
        only up0 exists and ``head`` is ignored.

        ``arm`` selects the per-arm heatmap head when num_arms > 1; ignored
        (must be 0) in the single-arm path.
        """
        if not self.use_modified_focal_loss:
            if self.num_arms == 1:
                assert arm == 0
                return self.up0(x)
            return self.up0_arms[arm](x)
        if head == "action":
            return self.up_action(x)
        if head == "grounding":
            return self.up_grounding(x)
        raise ValueError(f"Unknown head={head!r}; expected 'action' or 'grounding'.")

    def _paligemma_extract(self, img_rgb, language_goal):
        """Run PaliGemma on a (bs, num_img, 3, H, W) RGB tensor and return
        the per-view spatial token grid (bs, vlm_dim, num_img, 16, 16).

        Pure PaliGemma path with no memory / no head — split out so the
        same code is reused for the current frame (in-graph) and for the
        anchor / history frames (called inside torch.no_grad()).
        """
        bs = img_rgb.shape[0]
        # H20/Orion cuBLAS can SIGFPE on small-M bf16 PaliGemma forwards, which RMBench training hits when the
        # keyframe_gt history gather leaves only 1-3 valid memory frames. Pad the model input only and crop
        # tokens back below, so duplicated rows never reach the loss.
        min_bs = int(os.environ.get("PALIGEMMA_MIN_FORWARD_BS", "4"))
        pad_to_bs = max(bs, min_bs) if 0 < bs < min_bs else bs
        prompts = [text[0][0] for text in language_goal]
        images = [
            [MVT.trans_cuda_tensor_2_PIL(example) for example in examples]
            for examples in img_rgb
        ]
        assert len(prompts) == len(images)
        if pad_to_bs > bs:
            n_pad = pad_to_bs - bs
            prompts.extend([prompts[-1]] * n_pad)
            images.extend([images[-1]] * n_pad)

        # Prepend one <image> token per image to satisfy PaliGemmaProcessor.
        prompts = [("<image>" * len(imgs)) + p
                   for p, imgs in zip(prompts, images)]
        model_inputs = self.processor(
            text=prompts, images=images,
            return_tensors="pt", padding="longest",
        )
        model_inputs = model_inputs.to(self.model.dtype).to(self.model.device)
        # Only ``hidden_states[-1]`` is consumed; ``outputs.logits`` is never read. ``logits_to_keep=1`` runs
        # lm_head over a single position instead of the full sequence, and since output_hidden_states is
        # computed before the slice, the extracted tokens stay bit-identical.
        # lm_head is additionally STUBBED for this forward: even at logits_to_keep=1 the residual
        # (M=bs, K=2048) @ (2048, 257216) skinny GEMM deterministically SIGFPEs on this cluster's H20-3e
        # (Hopper sm_90 via Orion vGPU cuBLAS) in bf16/fp16, with no matmul-backend knob to avoid it. It only
        # bites at eval (bs=1); the logits are discarded anyway, so a shape-preserving stub replaces it.
        _lm = self.model.language_model.lm_head
        self.model.language_model.lm_head = _LMHeadStub()
        try:
            outputs = self.model(
                **model_inputs, output_hidden_states=True, logits_to_keep=1,
            )
        finally:
            self.model.language_model.lm_head = _lm

        hidden_states = outputs.hidden_states
        x = hidden_states[-1]

        # Extract the (256 * num_img) image tokens per sample.
        image_tokens = []
        for i in range(bs):
            current_ids = model_inputs["attention_mask"][i]
            current_output = x[i]
            non_zero_indices = torch.nonzero(current_ids != 0, as_tuple=True)[0]
            non_zero_output = current_output[non_zero_indices]
            assert non_zero_output.shape[0] > 256 * self.num_img
            non_zero_output = non_zero_output[:256 * self.num_img]
            image_tokens.append(non_zero_output)
        image_tokens = torch.stack(image_tokens)
        # (bs, vlm_dim, num_img, 16, 16)
        x = rearrange(
            image_tokens, 'b (c h1 h2) w -> b w c h1 h2',
            c=self.num_img, h1=self.num_pat_img, h2=self.num_pat_img,
        )
        return x

    def _apply_memory(self, x, stage, memory_imgs, memory_mask,
                      language_goal):
        """Inject spatial-anchor and (stage 1 only) temporal-episodic
        cross-attention into the per-view PaliGemma token grid.

        x:                (bs, vlm_dim, num_img, 16, 16)   — current tokens
        stage:            1 or 2
        memory_imgs:      dict (any of these may be missing / None):
                            "anchor":        (bs, V, channels, H, W) rendered
                                             anchor image (raw path)
                            "anchor_tokens": (bs, vlm_dim, V, H_p, W_p)
                                             pre-extracted anchor tokens
                                             (cache path, eval). When set,
                                             the raw-image PaliGemma forward
                                             on the anchor is SKIPPED.
                            "hist":          (bs, K, V, channels, H, W)
                                             rendered history images
                                             (raw path, training)
                            "hist_tokens":   (bs, K, vlm_dim, V, H_p, W_p)
                                             pre-extracted history tokens
                                             (cache path, eval).
        memory_mask:      dict with "anchor": (bs,) bool, "hist": (bs, K) bool

        Historical-action PE was removed; the temporal block now relies
        purely on visual KV + per-slot index.

        Returns x with memory contributions added (same shape).
        Anchor / history PaliGemma forwards (when running the raw path)
        and the cross-attn KV detach are both gated on
        ``self.memory_grad_through_tokens`` (set from
        ``memory.grad_through_tokens``, default True):
          * True: forwards build a graph, KVs are NOT detached, so
            gradient flows from the memory branch into PaliGemma.
          * False (legacy): forwards run under torch.no_grad() and KVs
            are detached, so the memory KV path contributes zero grad
            to PaliGemma.
        Eval is effectively unaffected since inference runs under
        no_grad anyway.
        """
        if not self.memory_enabled or memory_imgs is None:
            return x

        # Single switch shared by both raw-path PaliGemma forwards (anchor + history) and the post-extract detach.
        if self.memory_grad_through_tokens and self.training:
            _kv_ctx = contextlib.nullcontext()
            _kv_detach = lambda t: t            # noqa: E731 — identity
        else:
            _kv_ctx = torch.no_grad()
            _kv_detach = lambda t: t.detach()   # noqa: E731

        bs, vlm_dim, V, H_p, W_p = x.shape
        assert V == self.num_img
        N_q = H_p * W_p

        # (bs*V, N_q, dim) per-view query flatten.
        x_view = (
            x.permute(0, 2, 3, 4, 1)            # (bs, V, H_p, W_p, dim)
             .reshape(bs * V, N_q, vlm_dim)
        )
        # Memory block weights live in fp32 (default); inputs must match.
        x_view = x_view.to(torch.float32)

        anchor_img = memory_imgs.get("anchor", None)
        anchor_tokens = memory_imgs.get("anchor_tokens", None)
        hist_imgs = memory_imgs.get("hist", None)
        hist_tokens = memory_imgs.get("hist_tokens", None)
        anchor_mask = (memory_mask or {}).get("anchor", None)  # (bs,) bool
        hist_mask = (memory_mask or {}).get("hist", None)      # (bs, K) bool

        # ---- temporal block (stage 1 only) ----
        has_temporal_kv = (
            hist_tokens is not None or hist_imgs is not None
        )
        if (stage == 1 and hasattr(self, "mem_temporal_s1")
                and has_temporal_kv):
            if hist_tokens is not None:
                # Cache path (eval): tokens already extracted. (bs, K, vlm_dim, V, H_p, W_p)
                hist_tok_grid = hist_tokens
                K = hist_tok_grid.shape[1]
            else:
                # Raw path (training, or the eval first-frame fallback): run PaliGemma on the rendered history
                # images, wrapped in ``_kv_ctx`` (nullcontext when grad_through_tokens is True, else no_grad).
                # GATHER/SCATTER skips padded slots: under keyframe_gt every sample pads history to K slots but
                # only ``hist_mask`` slots are real, and a flat bs*K forward would pay the full 3B cost for the
                # fillers. Running only the valid frames and scattering token grids back to their original
                # (sample, slot) positions is numerically LOSSLESS — MemoryBlock excludes masked KV from the
                # softmax, and slot identity (slot_embed alignment) is preserved.
                K = hist_imgs.shape[1]
                hist_rgb_all = hist_imgs[:, :, :, 3:6, :, :].reshape(
                    bs * K, V, 3, hist_imgs.shape[-2], hist_imgs.shape[-1]
                )
                if hist_mask is not None:
                    mask_flat = hist_mask.reshape(-1).bool()
                else:
                    # No mask -> treat all slots valid (legacy semantics).
                    mask_flat = torch.ones(
                        bs * K, dtype=torch.bool, device=hist_rgb_all.device
                    )
                valid_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                n_valid = int(valid_idx.numel())

                if n_valid > 0:
                    hist_rgb = hist_rgb_all[valid_idx]   # (n_valid, V, 3, H, W)
                    # Slot-aligned language: flat index f -> sample f // K.
                    lang_hist = [
                        language_goal[int(f) // K]
                        for f in valid_idx.tolist()
                    ]
                    with _kv_ctx:
                        tok = self._paligemma_extract(hist_rgb, lang_hist)
                    # Scatter (n_valid, vlm_dim, V, H_p, W_p) back to flat (bs*K, ...). Index assignment is
                    # autograd-safe: grad flows to ``tok`` for valid slots only.
                    flat = tok.new_zeros(bs * K, vlm_dim, V, H_p, W_p)
                    flat[valid_idx] = tok
                else:
                    # Whole batch has empty history (e.g. all first-step): zeros -> fully masked, zero contribution.
                    flat = x.new_zeros(
                        bs * K, vlm_dim, V, H_p, W_p, dtype=torch.float32
                    )
                hist_tok_grid = flat.reshape(bs, K, vlm_dim, V, H_p, W_p)

            # (bs, K, vlm_dim, V, H_p, W_p) -> per-view-batched (bs*V, K, N_kv, vlm_dim).
            hist_tok = _kv_detach(
                hist_tok_grid.permute(0, 3, 1, 4, 5, 2).reshape(
                    bs * V, K, N_q, vlm_dim,
                ).to(torch.float32)
            )

            kv_mask = None
            if hist_mask is not None:
                # (bs, K) -> (bs*V, K)
                kv_mask = hist_mask.unsqueeze(1).expand(bs, V, K).reshape(
                    bs * V, K
                ).bool()

            x_view = self.mem_temporal_s1(
                x_view, hist_tok,
                kv_mask=kv_mask,
                num_views=V,
            )

        spatial_block = getattr(self, f"mem_spatial_s{stage}", None)
        has_anchor_kv = (anchor_tokens is not None or anchor_img is not None)
        if spatial_block is not None and has_anchor_kv:
            if anchor_tokens is not None:
                # Cache path (eval, stage 1 from step 1 onwards).
                anchor_tok_grid = anchor_tokens         # (bs, vlm_dim, V, H_p, W_p)
            else:
                # Raw path (training; and stage 2 every step, since its anchor zoom changes and cannot be cached).
                anchor_rgb = anchor_img[:, :, 3:6, :, :]
                with _kv_ctx:
                    anchor_tok_grid = self._paligemma_extract(
                        anchor_rgb, language_goal,
                    )
            # Stage-1 spatial uses CROSS-VIEW anchor KV: each current-view query attends to all V anchor views
            # (a V*N_q key bank). At stage 1 the camera geometry is fixed across the episode, so per-view
            # alignment is trivial and the useful signal is "did anything change anywhere relative to frame 0".
            # Stage-2 spatial keeps the per-view layout, because its anchor is re-rendered under the current
            # step's zoom and per-view geometric correspondence IS the signal it exploits.
            # No view PE on the anchor side — view identity is implicit in the rendered content.
            if stage == 1:
                # (bs, vlm_dim, V, H_p, W_p) -> (bs, V*N_q, vlm_dim)
                anchor_flat = anchor_tok_grid.permute(0, 2, 3, 4, 1).reshape(
                    bs, V * N_q, vlm_dim,
                )
                # Broadcast the V*N_q bank to every query view: (bs, 1, V*N_q, dim) -> (bs*V, V*N_q, dim).
                # The expand is stride-0 and the reshape forces a contiguous copy (~144 MB at V=3, bs=8, bf16).
                anchor_tok = _kv_detach(
                    anchor_flat.unsqueeze(1).expand(
                        bs, V, V * N_q, vlm_dim,
                    ).reshape(bs * V, V * N_q, vlm_dim).to(torch.float32)
                )
                n_kv = V * N_q
            else:
                # (bs, vlm_dim, V, H_p, W_p) -> (bs*V, N_q, vlm_dim) — per-view.
                anchor_tok = _kv_detach(
                    anchor_tok_grid.permute(0, 2, 3, 4, 1).reshape(
                        bs * V, N_q, vlm_dim,
                    ).to(torch.float32)
                )
                n_kv = N_q

            kv_mask = None
            if anchor_mask is not None:
                # (bs,) -> (bs*V, n_kv): per-sample validity broadcast over views and KV tokens. n_kv is
                # V*N_q at stage 1 (cross-view bank) and N_q at stage 2.
                kv_mask = anchor_mask.view(bs, 1, 1).expand(
                    bs, V, n_kv,
                ).reshape(bs * V, n_kv).bool()

            x_view = spatial_block(
                x_view, anchor_tok, kv_mask=kv_mask,
                num_views=(V if stage == 1 else 1),
            )

        # Reshape back: (bs*V, N_q, dim) -> (bs, dim, V, H_p, W_p).
        x_out = x_view.reshape(bs, V, H_p, W_p, vlm_dim).permute(
            0, 4, 1, 2, 3,
        ).contiguous()
        # Restore the incoming dtype so downstream heads see what they expect.
        x_out = x_out.to(x.dtype)
        return x_out

    def _forward_trunk(
        self,
        img,
        language_goal,
        stage,
        memory_imgs=None,
        memory_mask=None,
    ):
        """Shared trunk: PaliGemma extract + memory injection + reshape.

        Returns ``(x_heads, global_feat, mvt1_paligemma_tokens)`` where:
          * ``x_heads`` is (bs*num_img, vlm_dim, num_pat_img, num_pat_img)
            float32 — the per-view token grid the heatmap / feat heads read.
          * ``global_feat`` is (bs, num_img*vlm_dim) — max-pooled tokens,
            the first half of the feat_fc input.
          * ``mvt1_paligemma_tokens`` is the detached PRE-memory stage-1
            token snapshot for the eval MemoryBank (None outside stage 1 /
            when memory disabled).

        This trunk is SHARED across arms: at mvt1 it runs ONCE and both arms'
        heads read the same ``x_heads`` / ``global_feat``; at mvt2 it runs
        once per arm (each arm zooms to its own waypoint, so the rendered
        ``img`` and the per-arm spatial anchor differ).
        """
        bs, num_img, img_feat_dim, h, w = img.shape
        assert num_img == self.num_img
        assert h == w == self.img_size
        # Use only the rgb part (channels 3:6 inside the rendered tensor).
        img = img[:, :, 3:6, :, :]  # (bs, 3, 3, 224, 224)

        # Run PaliGemma on the current frame only. The old "concat current+anchor into one bs=2 forward"
        # optimization deterministically SIGFPEd inside cuBLAS at q_proj of Gemma decoder layer 0 on H20-3e +
        # torch 2.5.1+cu121 + bf16. For the same reason dual-arm must NEVER batch-concat the two arms — each
        # arm's stage-2 trunk is a separate call at the unchanged batch size.
        x = self._paligemma_extract(img, language_goal)

        # Snapshot of the PRE-memory current tokens for the eval bank to cache (only meaningful at stage 1).
        # Captured pre-memory so the cached "what step t saw" doesn't fold in step t's own memory contribution.
        if stage == 1 and self.memory_enabled:
            mvt1_paligemma_tokens = x.detach()
        else:
            mvt1_paligemma_tokens = None

        x = self._apply_memory(
            x, stage=stage,
            memory_imgs=memory_imgs,
            memory_mask=memory_mask,
            language_goal=language_goal,
        )

        # Per-sample max-pooled global feature for feat_fc.
        global_feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        global_feat = global_feat.view(bs, -1)

        # (bs*num_img, vlm_dim, num_pat_img, num_pat_img). Must be ``reshape``, not ``view``: with memory
        # enabled the strides after ``transpose(1, 2).clone()`` are no longer view-compatible for merging.
        x_heads = x.transpose(1, 2).reshape(
            bs * self.num_img, self.vlm_dim,
            self.num_pat_img, self.num_pat_img,
        ).to(torch.float32)

        return x_heads, global_feat, mvt1_paligemma_tokens

    def _forward_heads(
        self,
        x_heads,
        global_feat,
        wpt_local,
        rot_x_y,
        forward_no_feat,
        head="action",
        arm=0,
    ):
        """Per-arm heads on the shared trunk output.

        ``arm`` selects the per-arm heatmap head (always) and the per-arm
        rot/grip heads (when not ``forward_no_feat``); single-arm (arm==0)
        uses the flat heads. Returns the same ``out`` dict layout as the
        original forward (``trans`` + optionally ``feat_*``).
        """
        bs = global_feat.shape[0]
        h = w = self.img_size

        trans = self._heatmap_head(x_heads, head=head, arm=arm)
        trans = trans.view(bs, self.num_img, h, w)

        if forward_no_feat:
            return {"trans": trans}

        # Sample wpt at eval time using this stage's heatmap argmax.
        if not self.training:
            wpt_local = self.get_wpt(
                out={"trans": trans.clone().detach()},
                dyn_cam_info=None,
            )

        wpt_img = self.get_pt_loc_on_img(
            wpt_local.unsqueeze(1), dyn_cam_info=None,
        )
        wpt_img = wpt_img.reshape(bs * self.num_img, 2)

        if self.training:
            wpt_img = mvt_utils.add_uni_noi(
                wpt_img, self.wpt_img_aug * self.img_size
            )
        # At eval a predicted waypoint can project outside the image (a degenerate prediction); clamp it back
        # as in training, or the assertion below crashes on one out-of-bounds prediction.
        wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

        _wpt_img = wpt_img / self.img_patch_size
        _u = x_heads
        assert (0 <= _wpt_img.min() and _wpt_img.max() <= x_heads.shape[-1]), (
            _wpt_img, x_heads.shape
        )

        _wpt_img = _wpt_img.unsqueeze(1)
        _feat = select_feat_from_hm(_wpt_img, _u)[0]
        _feat = _feat.view(bs, -1)
        feat = torch.cat([global_feat, _feat], dim=-1)

        if self.rot_ver in (0, 2):
            # rot_ver==2 (6D regression) shares the single feat_fc path; the agent reads feat[:, :6] as the 6D rotation.
            feat_fc = self.feat_fc if self.num_arms == 1 else self.feat_fc_arms[arm]
            return {"feat": feat_fc(feat), "trans": trans}

        # rot_ver == 1: select this arm's rot/grip head set.
        if self.num_arms == 1:
            fc_ex_rot = self.feat_fc_ex_rot
            fc_init_bn = self.feat_fc_init_bn
            fc_x, fc_y, fc_z = self.feat_fc_x, self.feat_fc_y, self.feat_fc_z
        else:
            heads = self.feat_fc_rot_arms[arm]
            fc_ex_rot = heads["ex_rot"]
            fc_init_bn = heads["init_bn"]
            fc_x, fc_y, fc_z = heads["x"], heads["y"], heads["z"]

        feat_ex_rot = fc_ex_rot(feat)
        feat_rot = fc_init_bn(feat)
        feat_x = fc_x(feat_rot)

        if self.training:
            rot_x = rot_x_y[..., 0].view(bs, 1)
        else:
            rot_x = feat_x.argmax(dim=1, keepdim=True)
        rot_x_pe = self.feat_fc_pe(rot_x)
        feat_y = fc_y(feat_rot + rot_x_pe)

        if self.training:
            rot_y = rot_x_y[..., 1].view(bs, 1)
        else:
            rot_y = feat_y.argmax(dim=1, keepdim=True)
        rot_y_pe = self.feat_fc_pe(rot_y)
        feat_z = fc_z(feat_rot + rot_x_pe + rot_y_pe)
        return {
            "feat_ex_rot": feat_ex_rot,
            "feat_x": feat_x,
            "feat_y": feat_y,
            "feat_z": feat_z,
            "trans": trans,
        }

    def forward(
        self,
        img,
        wpt_local=None,
        rot_x_y=None,
        language_goal=None,
        forward_no_feat=False,
        head="action",
        stage=None,
        memory_imgs=None,
        memory_mask=None,
        arm=0,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param rot_x_y: (bs, 2)
        :param head: which head to route the heatmap through. Only
            consulted when use_modified_focal_loss=True. Default
            ``"action"``; pass ``"grounding"`` for the fv-mix branch.
        :param stage: 1 or 2 — selects which memory blocks fire. Defaults
            to ``1 if forward_no_feat else 2`` to match BridgeVLA's
            existing convention (mvt1 / stage1 has no feat_fc heads).
        :param memory_imgs / memory_mask: see :meth:`_apply_memory`.
            ``memory_imgs`` is None when memory is disabled or when the
            caller did not supply anchor/history renders for this stage.
        :param arm: per-arm head selector (num_arms > 1). The single-arm
            path keeps ``arm == 0`` and is byte-identical to before.

        Single-arm path = trunk once + one arm's heads. The dual-arm
        orchestration (shared trunk at mvt1, per-arm trunk at mvt2) lives in
        :class:`bridgevla.mvt.mvt.MVT`, which calls ``_forward_trunk`` /
        ``_forward_heads`` directly.
        """
        if stage is None:
            stage = 1 if forward_no_feat else 2

        x_heads, global_feat, mvt1_paligemma_tokens = self._forward_trunk(
            img, language_goal, stage,
            memory_imgs=memory_imgs, memory_mask=memory_mask,
        )
        out = self._forward_heads(
            x_heads, global_feat, wpt_local, rot_x_y,
            forward_no_feat=forward_no_feat, head=head, arm=arm,
        )
        # Expose pre-memory current-frame stage-1 tokens for the eval-time MemoryBank (absent otherwise).
        if mvt1_paligemma_tokens is not None:
            out["mvt1_paligemma_tokens"] = mvt1_paligemma_tokens
        # Keyframe discriminator on the POST-memory stage-1 token grid (single-arm path; the dual-arm path
        # runs it at the MVT level, and the two never both fire). Detached so the BCE trains only the
        # discriminator head, not the backbone. Only meaningful at stage 1.
        if stage == 1 and getattr(self, "discriminator_enabled", False):
            out["mem_logit"] = self.keyframe_disc(
                x_heads.detach(), num_views=self.num_img,
            )
        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """Estimate the (3D) waypoint from per-view heatmaps.

        Uses softmax over (h*w) — kept identical to the original BridgeVLA
        path.
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i: i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i: i + 1]
                if not (dyn_cam_info_itr[i] is None) else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if self.use_point_renderer:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None
        return pred_wpt

    def free_mem(self):
        """Free renderer memory after a batch (RVT renderer hook)."""
        print("Freeing up some memory")
        self.renderer.free_mem()
