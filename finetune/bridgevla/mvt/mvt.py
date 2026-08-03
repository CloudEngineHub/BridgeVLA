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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/mvt.py
Therefore, the code is also under the NVIDIA Source Code License


Architecture notes:
- Rotation / gripper / collision heads live inside mvt_single.py
  (feat_fc family).
- Eval-time per-view weighting always uses uniform 1/N averaging.
- View ablation: pass ``rend_three_views`` / ``rend_oblique_views``
  to the renderer. Default config is rend_three_views=True (original).
'''
import copy
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.mvt_single import MVT as MVTSingle
from bridgevla.mvt.config import get_cfg_defaults


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
        stage_two,
        st_sca,
        st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise,
        img_aug_2,
        renderer_device,
        load_pretrain=False,
        pretrain_path=None,
        flip_top_up=False,
        rotate_top_ccw90=False,
        renderer_img_sizes_w=2.0,
        splat_radius=0.012,
        splat_radius_mvt2=0.012,
        use_modified_focal_loss=False,
        use_view_logvar=False,
        memory=None,
        gradient_checkpointing=True,
        num_arms=1,
        predict_collision=True,
        feat_from_stage1=False,
    ):
        super().__init__()

        # NOTE: the renderer actually used is point_renderer.rvt_renderer.RVTBoxRenderer,
        # imported here as BoxRenderer, which shadows the BoxRenderer in ./renderer.py — the latter is unused.
        # The number of views is decided by RVTBoxRenderer's arguments (no_down=True, no_top=False by default):
        #   rend_oblique_views=True -> 3 views: oblique_a / oblique_b / oblique_c (54.7° magic angle)
        #   rend_three_views=True   -> 3 views: top / front / right (axis-aligned, the original BridgeVLA)
        #   both False              -> 5 views: top / front / back / left / right
        # Same class name, different implementation: when chasing view/heatmap counts, read RVTBoxRenderer, not ./renderer.py.
        from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer

        global BoxRenderer

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args["stage_two"]
        del args["st_sca"]
        del args["st_wpt_loc_aug"]
        del args["st_wpt_loc_inp_no_noise"]
        del args["img_aug_2"]
        del args["flip_top_up"]
        del args["rotate_top_ccw90"]
        del args["renderer_img_sizes_w"]
        del args["splat_radius"]
        del args["splat_radius_mvt2"]
        # feat_from_stage1 only affects the stage1/stage2 orchestration in
        # forward()/forward_dual() (which head emits the rot/grip/coll feat);
        # MVTSingle's head structure is unchanged, so keep it out of args.
        del args["feat_from_stage1"]
        if isinstance(renderer_img_sizes_w, (int, float)):
            _img_sizes_w = [float(renderer_img_sizes_w), float(renderer_img_sizes_w)]
        else:
            _img_sizes_w = list(renderer_img_sizes_w)
        # Exposed for the agent's stage-2 zoom-cube point count: the ortho
        # viewport spans ±img_sizes_w/2 (world units), NOT ±1, so the cube
        # actually visible to mvt2 has half-extent (img_sizes_w/2)/st_sca.
        self.renderer_img_sizes_w = _img_sizes_w
        # ``memory`` arrives as a yacs CfgNode; convert to a plain dict so
        # MVTSingle can pop optional keys without touching yacs internals.
        memory_cfg = None
        if memory is not None:
            memory_cfg = dict(memory)
        del args["memory"]
        args["memory_cfg"] = memory_cfg
        self.memory_cfg = memory_cfg or {}
        self.memory_enabled = bool(self.memory_cfg.get("enabled", False))

        self.rot_ver = rot_ver
        self.num_rot = num_rot
        # Dual-arm: num_arms > 1 triggers the per-arm orchestration in
        # forward(). num_arms / predict_collision stay in ``args`` and flow
        # to MVTSingle via **args (both accept them).
        self.num_arms = int(num_arms)
        self.predict_collision = bool(predict_collision)
        # When True, rot/grip/collision feat is produced by the stage-1
        # (coarse) heads and stage 2 emits only the translation heatmap.
        self.feat_from_stage1 = bool(feat_from_stage1)
        self.stage_two = stage_two
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.img_aug_2 = img_aug_2
        self.use_modified_focal_loss = use_modified_focal_loss
        self.use_view_logvar = use_view_logvar

        # for verifying the input
        self.feat_ver = feat_ver
        self.img_feat_dim = img_feat_dim

        assert not (rend_three_views and rend_oblique_views), \
            "rend_three_views (axis-aligned) and rend_oblique_views (magic-angle) are mutually exclusive"
        # mvt1 and mvt2 each use their own splat radius (both 0.012 by default). RMBench often makes mvt1
        # finer and leaves mvt2 at 0.012 (after trans_pc the stage-2 points are sparser, and a slightly coarser splat is more stable).
        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            oblique_views=rend_oblique_views,
            radius=splat_radius,
            with_depth=add_depth,
            flip_top_up=flip_top_up,
            rotate_top_ccw90=rotate_top_ccw90,
            img_sizes_w=_img_sizes_w,
        )
        self.renderer_mvt2 = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            oblique_views=rend_oblique_views,
            radius=splat_radius_mvt2,
            with_depth=add_depth,
            flip_top_up=flip_top_up,
            rotate_top_ccw90=rotate_top_ccw90,
            img_sizes_w=_img_sizes_w,
        )
        self.num_img = self.renderer.num_img
        self.img_size = img_size
        # MVTSingle keeps the mvt1 renderer for get_pt_loc_on_img / get_max_3d_frm_hm_cube;
        # both inference paths are based on the stage-1 fixed cameras and do not affect mvt2 rendering.
        self.mvt1 = MVTSingle(
            **args,
            renderer=self.renderer,
        )

    def get_pt_loc_on_img(self, pt, mvt1_or_mvt2, dyn_cam_info, out=None):
        """Project 3D pts to image-plane coords on stage 1 or 2 cameras."""
        assert len(pt.shape) == 3
        bs, _np, x = pt.shape
        assert x == 3

        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            assert out is None
            out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)
        else:
            assert self.stage_two
            assert out is not None
            assert out['wpt_local1'].shape == (bs, 3)
            pt, _ = mvt_utils.trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
            pt = pt.view(bs, _np, 3)
            out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, mvt1_or_mvt2, dyn_cam_info, y_q=None):
        """Estimate per-stage waypoint from heatmaps."""
        assert isinstance(mvt1_or_mvt2, bool)
        if mvt1_or_mvt2:
            wpt = self.mvt1.get_wpt(out, dyn_cam_info, y_q)
        else:
            assert self.stage_two
            wpt = self.mvt1.get_wpt(out["mvt2"], dyn_cam_info, y_q)
            wpt = out["rev_trans"](wpt)

        return wpt

    def render(self, pc, img_feat, img_aug, mvt1_or_mvt2, dyn_cam_info):
        assert isinstance(mvt1_or_mvt2, bool)

        mvt = self.mvt1
        renderer = self.renderer if mvt1_or_mvt2 else self.renderer_mvt2

        with torch.no_grad():
            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * len(pc)
            else:
                dyn_cam_info_itr = dyn_cam_info

            if mvt.add_corr:
                if mvt.norm_corr:
                    img = []
                    for _pc, _img_feat, _dyn_cam_info in zip(
                        pc, img_feat, dyn_cam_info_itr
                    ):
                        max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                        img.append(
                            renderer(
                                _pc,
                                torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                fix_cam=True,
                                dyn_cam_info=(_dyn_cam_info,)
                                if not (_dyn_cam_info is None) else None,
                            ).unsqueeze(0)
                        )
                else:
                    img = [
                        renderer(
                            _pc,
                            torch.cat((_pc, _img_feat), dim=-1),
                            fix_cam=True,
                            dyn_cam_info=(_dyn_cam_info,)
                            if not (_dyn_cam_info is None) else None,
                        ).unsqueeze(0)
                        for (_pc, _img_feat, _dyn_cam_info) in zip(
                            pc, img_feat, dyn_cam_info_itr
                        )
                    ]
            else:
                img = [
                    renderer(
                        _pc,
                        _img_feat,
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None) else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]

        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3)

        if mvt.add_corr:
            mvt.img = img[:, :, 3:].clone().detach()
        else:
            mvt.img = img.clone().detach()

        if img_aug != 0:
            stdv = img_aug * torch.rand(1, device=img.device)
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)

        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device)
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )

        return img

    def verify_inp(self, pc, img_feat, img_aug, wpt_local, rot_x_y):
        bs = len(pc)
        assert bs == len(img_feat)

        if not self.training:
            assert img_aug == 0

        if self.training:
            assert (not self.feat_ver == 1) or (not wpt_local is None)

            if self.rot_ver in (0, 2):
                # rot_ver==2 (6D regression) has no autoregressive rot_x_y
                # conditioning (that is rot_ver==1 only).
                assert rot_x_y is None, f"rot_x_y={rot_x_y}"
            elif self.rot_ver == 1:
                assert rot_x_y.shape == (bs, 2), f"rot_x_y.shape={rot_x_y.shape}"
                assert (rot_x_y >= 0).all() and (rot_x_y < self.num_rot).all()
            else:
                assert False

        for _pc, _img_feat in zip(pc, img_feat):
            np_, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np_ == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if not (wpt_local is None):
            bs5, x6 = wpt_local.shape
            assert bs == bs5
            assert x6 == 3, f"Does not support wpt_local of shape {wpt_local.shape}"

        if self.training:
            assert (not self.stage_two) or (not wpt_local is None)

    def forward(
        self,
        pc=None,
        img_feat=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        language_goal=None,
        mode="action",
        viz_force_gt_stage2=False,
        memory_inputs=None,
        **kwargs,
    ):
        """
        :param mode: only "action" is supported — the full PC-rendered
            forward.
        :param memory_inputs: optional dict (only consulted when
            ``self.memory_enabled``) with keys::

                anchor_pc       list[(N_i, 3)]    per-sample anchor PC
                anchor_img_feat list[(N_i, F)]    per-sample anchor feats
                anchor_mask     (bs,) bool        True = anchor valid
                hist_pc         list[ list[(N_i, 3)] ]    K * bs PCs
                hist_img_feat   list[ list[(N_i, F)] ]    K * bs feats
                hist_mask       (bs, K) bool      per-slot validity

            Historical actions are NOT consumed (``MemoryBlock.action_proj``
            was removed; temporal memory is purely visual + per-slot index
            PE). Any legacy ``hist_action`` field that may still appear in
            old replay samples is silently ignored.

            stage1 receives anchor + hist; stage2 receives anchor zoom-
            synchronised with the current frame (same trans_pc(loc, sca)).
            Anchor / history rendering happens here under no_grad.
        """
        if mode != "action":
            raise ValueError(f"Unknown mode={mode!r}; expected 'action'.")

        if self.num_arms > 1:
            return self.forward_dual(
                pc=pc, img_feat=img_feat, img_aug=img_aug,
                wpt_local=wpt_local, rot_x_y=rot_x_y,
                language_goal=language_goal,
                viz_force_gt_stage2=viz_force_gt_stage2,
                memory_inputs=memory_inputs, **kwargs,
            )

        self.verify_inp(
            pc=pc, img_feat=img_feat, img_aug=img_aug,
            wpt_local=wpt_local, rot_x_y=rot_x_y,
        )

        # ---- Memory inputs (anchor + history) for stage 1. ----
        # Two paths converge on the same MVTSingle interface:
        #   * Cache path (eval, step >= 1): the agent passes the
        #     pre-extracted PaliGemma tokens for anchor and history via
        #     ``memory_inputs["anchor_mvt1_tokens"]`` /
        #     ``memory_inputs["hist_mvt1_tokens"]``. Stage-1 cameras are
        #     fixed, so these tokens never change within an episode and
        #     re-running PaliGemma every step is wasted compute.
        #   * Raw path (training, or eval first frame): tokens are NOT
        #     supplied; we render anchor / hist images with the same mvt1
        #     camera geometry as the current frame and let MVTSingle run
        #     PaliGemma on them under no_grad.
        memory_active = bool(self.memory_enabled and memory_inputs is not None)
        anchor_img_mvt1 = None
        anchor_tokens_mvt1 = None
        hist_img_mvt1 = None
        hist_tokens_mvt1 = None
        anchor_mask = None
        hist_mask = None
        if memory_active:
            anchor_pc_raw = memory_inputs.get("anchor_pc")
            anchor_feat_raw = memory_inputs.get("anchor_img_feat")
            anchor_mask = memory_inputs.get("anchor_mask")
            anchor_tokens_mvt1 = memory_inputs.get("anchor_mvt1_tokens")
            hist_pc_raw = memory_inputs.get("hist_pc")
            hist_feat_raw = memory_inputs.get("hist_img_feat")
            hist_mask = memory_inputs.get("hist_mask")
            hist_tokens_mvt1 = memory_inputs.get("hist_mvt1_tokens")
            with torch.no_grad():
                # Anchor: render only when no cached tokens were supplied.
                if anchor_tokens_mvt1 is None and anchor_pc_raw is not None:
                    anchor_img_mvt1 = self.render(
                        pc=anchor_pc_raw, img_feat=anchor_feat_raw,
                        img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
                    )
                # History: render only when no cached tokens were supplied.
                if (hist_tokens_mvt1 is None
                        and hist_pc_raw is not None and len(hist_pc_raw) > 0):
                    K = len(hist_pc_raw)
                    bs = len(hist_pc_raw[0])
                    per_slot = []
                    for k in range(K):
                        per_slot.append(self.render(
                            pc=hist_pc_raw[k], img_feat=hist_feat_raw[k],
                            img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
                        ))
                    # Each entry is (bs, V, C, H, W); stack to (bs, K, V, C, H, W).
                    hist_img_mvt1 = torch.stack(per_slot, dim=1)
                    assert hist_img_mvt1.shape[0] == bs and hist_img_mvt1.shape[1] == K

        with torch.no_grad():
            if self.training and (self.img_aug_2 != 0):
                for i, x in enumerate(img_feat):
                    stdv = self.img_aug_2 * torch.rand(1, device=x.device)
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    img_feat[i] = x + noise

            img = self.render(
                pc=pc, img_feat=img_feat, img_aug=img_aug,
                mvt1_or_mvt2=True, dyn_cam_info=None,
            )

        if self.training:
            wpt_local_stage_one = wpt_local.clone().detach()
        else:
            wpt_local_stage_one = wpt_local

        # Stage 1: original BridgeVLA path -> trans only (no feat).
        # When the cache path is used, ``anchor_tokens_mvt1`` /
        # ``hist_tokens_mvt1`` are populated and the matching raw-image
        # entries are None — MVTSingle picks the cache path automatically
        # (see ``_apply_memory``).
        s1_memory_imgs = None
        s1_memory_mask = None
        if memory_active:
            s1_memory_imgs = {
                "anchor": anchor_img_mvt1,
                "anchor_tokens": anchor_tokens_mvt1,
                "hist": hist_img_mvt1,
                "hist_tokens": hist_tokens_mvt1,
            }
            s1_memory_mask = {
                "anchor": anchor_mask,
                "hist": hist_mask,
            }
        out = self.mvt1(
            img=img,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            language_goal=language_goal,
            forward_no_feat=not self.feat_from_stage1,
            head="action",
            stage=1,
            memory_imgs=s1_memory_imgs,
            memory_mask=s1_memory_mask,
            **kwargs,
        )
        out["mvt1_ori_img"] = img.clone().detach()

        if self.stage_two:
            with torch.no_grad():
                # `viz_force_gt_stage2`: eval-mode visualization hook — when True
                # and a GT `wpt_local` is supplied, take the SAME stage-2 zoom
                # center as training (GT + `st_wpt_loc_aug` noise) instead of
                # mvt1's prediction. Lets viz show how far mvt2 drifts under
                # the exact crop it was trained on.
                use_gt_zoom = self.training or (
                    viz_force_gt_stage2 and wpt_local is not None
                )
                if use_gt_zoom:
                    assert wpt_local is not None
                    wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
                        wpt_local_stage_one.clone().detach(),
                        2 * self.st_wpt_loc_aug,
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca,
                    )
                    if self.st_wpt_loc_inp_no_noise:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one_noisy,
                            sca=self.st_sca,
                        )
                    else:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local, loc=wpt_local_stage_one,
                            sca=self.st_sca,
                        )
                else:
                    # Eval: zoom into mvt1's predicted waypoint.
                    wpt_local = self.get_wpt(
                        out, y_q=None, mvt1_or_mvt2=True,
                        dyn_cam_info=None,
                    )
                    pc, rev_trans = mvt_utils.trans_pc(
                        pc, loc=wpt_local, sca=self.st_sca,
                    )
                    wpt_local_stage_one_noisy = wpt_local
                    wpt_local2 = None

                img = self.render(
                    pc=pc, img_feat=img_feat, img_aug=img_aug,
                    mvt1_or_mvt2=False, dyn_cam_info=None,
                )

                # Anchor zoom-synchronisation: apply the SAME trans_pc to
                # the anchor PC so anchor and current share the zoom-local
                # camera coordinates exactly. ``rev_trans`` from this call
                # is discarded — only the anchor render is used downstream
                # and the current frame's rev_trans (above) is what maps
                # waypoints back to world coords.
                anchor_img_mvt2 = None
                if memory_active and memory_inputs.get("anchor_pc") is not None:
                    anchor_pc_zoom, _ = mvt_utils.trans_pc(
                        memory_inputs["anchor_pc"],
                        loc=wpt_local_stage_one_noisy,
                        sca=self.st_sca,
                    )
                    anchor_img_mvt2 = self.render(
                        pc=anchor_pc_zoom,
                        img_feat=memory_inputs["anchor_img_feat"],
                        img_aug=0, mvt1_or_mvt2=False, dyn_cam_info=None,
                    )

            # Stage 2: trans + feat_x/y/z + feat_ex_rot.
            s2_memory_imgs = None
            s2_memory_mask = None
            if memory_active and anchor_img_mvt2 is not None:
                # Stage 2 deliberately has no temporal block — only spatial
                # anchor at the current zoom.
                s2_memory_imgs = {"anchor": anchor_img_mvt2, "hist": None}
                s2_memory_mask = {"anchor": anchor_mask, "hist": None}
            out_mvt2 = self.mvt1(
                img=img,
                wpt_local=wpt_local2,
                rot_x_y=rot_x_y,
                language_goal=language_goal,
                forward_no_feat=self.feat_from_stage1,
                head="action",
                stage=2,
                memory_imgs=s2_memory_imgs,
                memory_mask=s2_memory_mask,
                **kwargs,
            )

            out["wpt_local1"] = wpt_local_stage_one_noisy
            out["rev_trans"] = rev_trans
            out["mvt2"] = out_mvt2
            out["mvt2_ori_img"] = img.clone().detach()
            if anchor_img_mvt2 is not None:
                out["mvt2_anchor_ori_img"] = anchor_img_mvt2.clone().detach()

        return out

    def forward_dual(
        self,
        pc=None,
        img_feat=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        language_goal=None,
        viz_force_gt_stage2=False,
        memory_inputs=None,
        **kwargs,
    ):
        """Dual-arm forward (num_arms > 1).

        Per the confirmed RMBench design:
          * mvt1 (GLOBAL): ONE shared tri-view render, ONE shared PaliGemma
            trunk, SHARED spatial-anchor-s1 + temporal-history-s1 memory.
            Only the heatmap head branches per arm -> per-arm stage-1
            waypoint = that arm's independent stage-2 zoom center.
          * mvt2 (ZOOM): per arm — each arm zooms the ORIGINAL point cloud to
            its own waypoint, renders its own tri-view, runs its own PaliGemma
            trunk, and (if memory) cross-attends its own zoom-synchronised
            anchor. Per-arm heatmap + rot/grip heads.

        ``wpt_local`` / ``rot_x_y`` are per-arm lists of length num_arms at
        training (GT), or None at eval. Returns ``out`` with:
          out["per_arm"]            list[num_arms] of single-arm-style out
                                    dicts (each has trans, mvt2, wpt_local1,
                                    rev_trans, mvt*_ori_img) so the agent can
                                    reuse get_q / get_action_trans / get_pred
                                    per arm unchanged.
          out["mvt1_paligemma_tokens"]  shared stage-1 token snapshot (one
                                    trunk) for the eval MemoryBank push.
        """
        n_arms = self.num_arms
        # Normalize per-arm inputs to length-n_arms lists.
        if wpt_local is None:
            wpt_local_arms = [None] * n_arms
        else:
            assert len(wpt_local) == n_arms, (len(wpt_local), n_arms)
            wpt_local_arms = list(wpt_local)
        if rot_x_y is None:
            rot_x_y_arms = [None] * n_arms
        else:
            assert len(rot_x_y) == n_arms, (len(rot_x_y), n_arms)
            rot_x_y_arms = list(rot_x_y)

        bs = len(pc)
        assert bs == len(img_feat)

        # ---- Shared mvt1 memory (anchor + history), identical for arms. ----
        memory_active = bool(self.memory_enabled and memory_inputs is not None)
        anchor_img_mvt1 = None
        anchor_tokens_mvt1 = None
        hist_img_mvt1 = None
        hist_tokens_mvt1 = None
        anchor_mask = None
        hist_mask = None
        if memory_active:
            anchor_pc_raw = memory_inputs.get("anchor_pc")
            anchor_feat_raw = memory_inputs.get("anchor_img_feat")
            anchor_mask = memory_inputs.get("anchor_mask")
            anchor_tokens_mvt1 = memory_inputs.get("anchor_mvt1_tokens")
            hist_pc_raw = memory_inputs.get("hist_pc")
            hist_feat_raw = memory_inputs.get("hist_img_feat")
            hist_mask = memory_inputs.get("hist_mask")
            hist_tokens_mvt1 = memory_inputs.get("hist_mvt1_tokens")
            with torch.no_grad():
                if anchor_tokens_mvt1 is None and anchor_pc_raw is not None:
                    anchor_img_mvt1 = self.render(
                        pc=anchor_pc_raw, img_feat=anchor_feat_raw,
                        img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
                    )
                if (hist_tokens_mvt1 is None
                        and hist_pc_raw is not None and len(hist_pc_raw) > 0):
                    K = len(hist_pc_raw)
                    per_slot = [
                        self.render(
                            pc=hist_pc_raw[k], img_feat=hist_feat_raw[k],
                            img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
                        )
                        for k in range(K)
                    ]
                    hist_img_mvt1 = torch.stack(per_slot, dim=1)

        # ---- Shared img_feat aug + shared mvt1 render (ONCE). ----
        with torch.no_grad():
            if self.training and (self.img_aug_2 != 0):
                for i, x in enumerate(img_feat):
                    stdv = self.img_aug_2 * torch.rand(1, device=x.device)
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    img_feat[i] = x + noise
            img1 = self.render(
                pc=pc, img_feat=img_feat, img_aug=img_aug,
                mvt1_or_mvt2=True, dyn_cam_info=None,
            )

        s1_memory_imgs = None
        s1_memory_mask = None
        if memory_active:
            s1_memory_imgs = {
                "anchor": anchor_img_mvt1,
                "anchor_tokens": anchor_tokens_mvt1,
                "hist": hist_img_mvt1,
                "hist_tokens": hist_tokens_mvt1,
            }
            s1_memory_mask = {"anchor": anchor_mask, "hist": hist_mask}

        # ---- Shared mvt1 trunk (ONE PaliGemma + memory forward). ----
        x_heads1, global_feat1, mvt1_tokens = self.mvt1._forward_trunk(
            img1, language_goal, stage=1,
            memory_imgs=s1_memory_imgs, memory_mask=s1_memory_mask,
        )
        # ``*_ori_img`` are detached RGB snapshots consumed ONLY by the eval-time
        # visualizer (visualize.py runs under agent.eval()). They are never read
        # on the training path, so skip the clones while training to avoid
        # holding several (bs, V, C, H, W) bf16 copies through backward.
        keep_ori_img = not self.training
        img1_det = img1.clone().detach() if keep_ori_img else None

        # ---- Keyframe discriminator on the shared post-memory stage-1
        # tokens (scene-level, arm-independent). logit -> BCE at train,
        # sigmoid gate for MemoryBank.push at eval. ----
        mem_logit = None
        if getattr(self.mvt1, "discriminator_enabled", False):
            # Detach so the discriminator BCE trains ONLY its own params and
            # never backprops into the shared mvt1 trunk / memory blocks. The
            # disc input (x_heads1) is the exact tensor the per-arm heatmap
            # heads consume, so an un-detached disc gradient (amplified by
            # pos_weight) would reshape the shared stage-1 features both arms
            # depend on, corrupting the memory-dependent (e.g. left-arm)
            # phases. Eval gating is unaffected (inference runs under no_grad).
            mem_logit = self.mvt1.keyframe_disc(
                x_heads1.detach(), num_views=self.mvt1.num_img,
            )

        # Keep the ORIGINAL cloud — every arm zooms from it (never from a
        # previously-zoomed cloud).
        pc_orig = pc

        per_arm_out = []
        for a in range(n_arms):
            # Stage-1 heatmap for arm a (shared trunk, per-arm head).
            # feat_from_stage1=True: stage-1 (coarse, shared trunk) heads ALSO
            # emit the rot/grip/coll feat — stage-1 global max-pool + this arm's
            # stage-1 waypoint local patch (per-arm feat_fc head). Otherwise
            # stage 1 emits only the coarse heatmap (original behavior).
            out_a = self.mvt1._forward_heads(
                x_heads1, global_feat1,
                wpt_local=wpt_local_arms[a], rot_x_y=rot_x_y_arms[a],
                forward_no_feat=not self.feat_from_stage1, head="action", arm=a,
            )
            if keep_ori_img:
                out_a["mvt1_ori_img"] = img1_det

            if not self.stage_two:
                per_arm_out.append(out_a)
                continue

            # Stage-1 zoom center for this arm.
            if self.training:
                wpt_local_a = wpt_local_arms[a]
                wpt_local_stage_one = wpt_local_a.clone().detach()
            else:
                wpt_local_a = wpt_local_arms[a]
                wpt_local_stage_one = wpt_local_a

            with torch.no_grad():
                use_gt_zoom = self.training or (
                    viz_force_gt_stage2 and wpt_local_a is not None
                )
                if use_gt_zoom:
                    assert wpt_local_a is not None
                    wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
                        wpt_local_stage_one.clone().detach(),
                        2 * self.st_wpt_loc_aug,
                    )
                    pc_zoom, rev_trans = mvt_utils.trans_pc(
                        pc_orig, loc=wpt_local_stage_one_noisy, sca=self.st_sca,
                    )
                    if self.st_wpt_loc_inp_no_noise:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local_a, loc=wpt_local_stage_one_noisy,
                            sca=self.st_sca,
                        )
                    else:
                        wpt_local2, _ = mvt_utils.trans_pc(
                            wpt_local_a, loc=wpt_local_stage_one,
                            sca=self.st_sca,
                        )
                else:
                    # Eval: zoom into arm a's predicted stage-1 waypoint.
                    wpt_pred_a = self.get_wpt(
                        out_a, y_q=None, mvt1_or_mvt2=True, dyn_cam_info=None,
                    )
                    pc_zoom, rev_trans = mvt_utils.trans_pc(
                        pc_orig, loc=wpt_pred_a, sca=self.st_sca,
                    )
                    wpt_local_stage_one_noisy = wpt_pred_a
                    wpt_local2 = None

                img2 = self.render(
                    pc=pc_zoom, img_feat=img_feat, img_aug=img_aug,
                    mvt1_or_mvt2=False, dyn_cam_info=None,
                )

                # Per-arm anchor zoom-synchronisation (anchor differs per arm
                # because each arm zooms to its own center).
                anchor_img_mvt2 = None
                if memory_active and memory_inputs.get("anchor_pc") is not None:
                    anchor_pc_zoom, _ = mvt_utils.trans_pc(
                        memory_inputs["anchor_pc"],
                        loc=wpt_local_stage_one_noisy, sca=self.st_sca,
                    )
                    anchor_img_mvt2 = self.render(
                        pc=anchor_pc_zoom,
                        img_feat=memory_inputs["anchor_img_feat"],
                        img_aug=0, mvt1_or_mvt2=False, dyn_cam_info=None,
                    )

            s2_memory_imgs = None
            s2_memory_mask = None
            if memory_active and anchor_img_mvt2 is not None:
                s2_memory_imgs = {"anchor": anchor_img_mvt2, "hist": None}
                s2_memory_mask = {"anchor": anchor_mask, "hist": None}

            # Stage-2 per-arm trunk (separate PaliGemma forward per arm —
            # NEVER batch-concat arms; see SIGFPE note in MVTSingle).
            x_heads2, global_feat2, _ = self.mvt1._forward_trunk(
                img2, language_goal, stage=2,
                memory_imgs=s2_memory_imgs, memory_mask=s2_memory_mask,
            )
            # feat_from_stage1=True: stage 2 emits ONLY the fine translation
            # heatmap (forward_no_feat=True); the rot/grip/coll feat is taken
            # from stage 1 above. Default: stage 2 emits trans + feat.
            out_mvt2 = self.mvt1._forward_heads(
                x_heads2, global_feat2,
                wpt_local=wpt_local2, rot_x_y=rot_x_y_arms[a],
                forward_no_feat=self.feat_from_stage1, head="action", arm=a,
            )

            out_a["wpt_local1"] = wpt_local_stage_one_noisy
            out_a["rev_trans"] = rev_trans
            out_a["mvt2"] = out_mvt2
            if keep_ori_img:
                out_a["mvt2_ori_img"] = img2.clone().detach()
                if anchor_img_mvt2 is not None:
                    out_a["mvt2_anchor_ori_img"] = anchor_img_mvt2.clone().detach()
            per_arm_out.append(out_a)

        out = {"per_arm": per_arm_out}
        if keep_ori_img:
            out["mvt1_ori_img"] = img1_det
        # Shared stage-1 token snapshot for the eval memory bank (one trunk).
        if mvt1_tokens is not None:
            out["mvt1_paligemma_tokens"] = mvt1_tokens
        if mem_logit is not None:
            out["mem_logit"] = mem_logit
        return out


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
