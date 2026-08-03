'''
Episodic memory for BridgeVLA++ (GemBench branch).

Two memory mechanisms hang off the PaliGemma image-token grid:

  * Spatial Anchor (stage1 + stage2)
        Episode-frozen tokens from the unoccluded first frame, queried via
        cross-attention.
          - Stage 1: CROSS-VIEW. Each current-view query attends to the
            concatenation of ALL V anchor views' tokens (V*N_q keys).
            Stage-1 cameras are fixed across the episode, so per-view
            geometric alignment is trivial; the useful signal is "did
            anything anywhere in the scene change vs frame 0".
          - Stage 2: PER-VIEW (legacy). View v query only sees view v
            anchor; the anchor PC is re-rendered with the SAME
            trans_pc(loc=wpt, sca=st_sca) as the current frame, so
            anchor and current share the zoom-local camera coordinates
            exactly — per-view geometric correspondence IS the signal
            stage 2 exploits.
        No view PE is injected on the anchor side at either stage; view
        identity is implicit in the rendered image content.

  * Temporal Episodic (stage1 only)
        K-slot temporal memory of keyframe tokens. A per-slot learned
        embedding is added to the KV before to_kv. Historical-action PE
        has been removed — temporal memory is purely visual + slot index.
        stage2 deliberately has no temporal block.

        Two slot conventions, chosen by ``MemoryBank(select=...)`` (eval) /
        ``memory_select`` (training):
          - "sliding" (legacy): every slot holds an admitted keyframe;
            slot 0 = most recent.
          - "keyframe_gt" (RMBench): slot 0 / slot 1 ALWAYS hold the two
            most-recent EXECUTED keyframes (t-1, t-2) regardless of whether
            they are subtask boundaries; slots 2..K-1 hold GT subtask-
            boundary "memory keyframes" strictly older than t-2 (j <= t-3),
            most-recent first. A keyframe judged a subtask boundary is only
            admitted into the memory portion two steps later (once it has
            scrolled out of the two recency slots), so it never occupies a
            recency slot AND a memory slot simultaneously — no duplication.

The anchor / history PaliGemma forwards are gated on
``memory.grad_through_tokens`` (default True): when True they build a
graph and their KV tokens are NOT detached before cross-attn, so
gradient flows from the memory branch back into PaliGemma; when False
the legacy behavior is used (no_grad forwards + detached KVs, zero
grad to PaliGemma from the memory KV path). The block's residual exits
use the default initialisation of their submodules (Attention /
FeedForward), so the memory branch contributes a non-zero perturbation
from step 0.

'''
import torch
import torch.utils.checkpoint
from torch import nn

from bridgevla.mvt.attn import Attention, FeedForward, PreNorm


class MemoryBlock(nn.Module):
    """N-layer memory module. Each layer = cross-attn -> self-attn -> ffn.

    Inputs to ``forward``:
      x        (bs*V, N_q, dim)            current-frame query tokens
      ctx      spatial:  (bs*V, N_kv, dim) detached KV
                         (caller controls layout: at stage-1 spatial the
                          caller broadcasts V views' anchor tokens so
                          N_kv = V*N_q; at stage-2 spatial it's the
                          per-view N_kv = N_q. The block itself is shape-
                          agnostic — it just attends over whatever KV it
                          is given.)
               temporal: (bs*V, K, N_kv, dim) detached KV (per slot)
      kv_mask  spatial:  (bs*V, N_kv) bool, True=valid (or None for all-valid)
               temporal: (bs*V, K)    bool, True=valid slot

    Multi-layer behaviour: the same memory KV is reused at every layer
    (Perceiver-IO style), but each layer has its own Q/K/V/output
    projections + self-attn + FFN. Time encoding (slot_embed for
    temporal) is computed ONCE at the block level and shared across
    layers — the KV identity doesn't change between layers.
    """

    def __init__(
        self,
        dim: int = 2048,
        heads: int = 8,
        dim_head: int = 128,
        num_layers: int = 2,
        kind: str = "spatial",
        K_temporal: int = 2,
        ffn_mult: int = 2,
        use_fast: bool = False,
    ):
        super().__init__()
        assert kind in ("spatial", "temporal"), kind
        assert num_layers >= 1, num_layers
        self.kind = kind
        self.dim = dim
        self.num_layers = int(num_layers)
        self.K_temporal = K_temporal
        # Activation checkpointing toggle (set by MVTSingle to mirror the
        # PaliGemma ``gradient_checkpointing`` flag). LOSS-LESS: these blocks
        # contain no dropout / randomness (Attention dropout_p=0, no FFN
        # dropout, slot_embed is a lookup), so the backward recompute is
        # bit-identical to the cached forward. It only trades compute for
        # peak memory — the large fp32 cross-attn score matrices
        # (temporal KV = K*N_q tokens) are no longer retained until backward.
        self.grad_checkpoint = True

        # Per-layer cross / self / ffn. Each layer has its own QKV/output
        # projections; they don't share parameters across depth.
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross": PreNorm(
                    dim,
                    Attention(query_dim=dim, context_dim=dim,
                              heads=heads, dim_head=dim_head,
                              use_fast=use_fast),
                    context_dim=dim,
                ),
                "self_attn": PreNorm(
                    dim,
                    Attention(query_dim=dim, heads=heads,
                              dim_head=dim_head, use_fast=use_fast),
                ),
                "ffn": PreNorm(dim, FeedForward(dim, mult=ffn_mult)),
            })
            for _ in range(self.num_layers)
        ])

        # Time encoding (temporal kind only) — block-level, shared across
        # layers. The same KV is reused at every depth, so it makes no
        # sense to re-add slot PE at every layer. Historical-action PE
        # has been removed; temporal memory now relies purely on visual
        # KV plus per-slot index.
        if kind == "temporal":
            self.slot_embed = nn.Embedding(K_temporal, dim)

        # NOTE: residual-exit projections (cross.to_out, self_attn.to_out,
        # FFN last linear) are left at the default initialisation of their
        # submodules. The block therefore perturbs the backbone from step
        # 0 rather than starting as identity.

    def forward(
        self,
        x,
        ctx,
        kv_mask=None,
        num_views: int = 1,
    ):
        # Wrap the (heavy, fp32) memory forward in activation checkpointing
        # during training so its cross/self-attn score matrices are recomputed
        # in backward instead of being held through the whole step. ``ctx`` is
        # passed THROUGH the checkpoint so its grad (back into PaliGemma when
        # grad_through_tokens=True) is preserved; the memory block's own params
        # always require grad, so use_reentrant=False recomputes correctly even
        # when x/ctx don't require grad (Stage-1 frozen-PaliGemma phase).
        # Eval runs under no_grad and stores nothing, so checkpoint is skipped.
        if (self.grad_checkpoint and self.training
                and torch.is_grad_enabled()):
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, ctx, kv_mask, num_views,
                use_reentrant=False,
            )
        return self._forward_impl(x, ctx, kv_mask, num_views)

    def _forward_impl(
        self,
        x,
        ctx,
        kv_mask=None,
        num_views: int = 1,
    ):
        if self.kind == "temporal":
            assert ctx.dim() == 4, (
                f"temporal ctx should be (bs*V, K, N_kv, dim); got {ctx.shape}"
            )
            bs_V, K, N_kv, dim = ctx.shape
            assert K == self.K_temporal, (K, self.K_temporal)

            # Per-slot learned embedding, broadcast over (bs*V, K, N_kv, dim).
            # Historical action PE was removed; only the slot index PE
            # is injected so the cross-attn can tell slots apart.
            slot_pe = self.slot_embed(
                torch.arange(K, device=x.device)
            )  # (K, dim)
            slot_pe = slot_pe.view(1, K, 1, dim)

            ctx = ctx + slot_pe                            # add BEFORE to_kv
            ctx = ctx.reshape(bs_V, K * N_kv, dim)

            if kv_mask is not None:
                assert kv_mask.shape == (bs_V, K), (kv_mask.shape, bs_V, K)
                kv_mask = kv_mask.unsqueeze(-1).expand(
                    bs_V, K, N_kv
                ).reshape(bs_V, K * N_kv)
        else:
            assert ctx.dim() == 3, (
                f"spatial ctx should be (bs*V, N_kv, dim); got {ctx.shape}"
            )
            bs_V = ctx.shape[0]

        # Guard: rows whose KV is fully masked would NaN out the softmax.
        # Trick: temporarily allow all KV in those rows, then zero all
        # memory-layer residual contributions post-hoc.
        if kv_mask is not None:
            row_has_valid = kv_mask.any(dim=-1, keepdim=True)  # (bs_V, 1)
            attn_mask = kv_mask | (~row_has_valid)             # all-True for empty rows
            residual_gate = row_has_valid.to(x.dtype).unsqueeze(-1)
        else:
            row_has_valid = None
            attn_mask = None
            residual_gate = None

        # Iterate over layers. Same KV every layer; each layer has its own
        # Q/K/V/out projections + self-attn + FFN.
        for layer in self.layers:
            cross_out = layer["cross"](x, context=ctx, mask=attn_mask)
            if residual_gate is not None:
                # Multiply the residual contributions to zero for samples
                # with no valid KV (e.g. first-step temporal).
                cross_out = cross_out * residual_gate
            x = x + cross_out

            # Self-attn: when num_views > 1, merge all V views into one
            # sequence so tokens from different views can attend to each
            # other: (bs*V, N_q, dim) -> (bs, V*N_q, dim) -> back.
            # residual_gate is per-sample (same value for all V rows of
            # the same sample), so we take the first-view slice after
            # reshaping.
            if num_views > 1:
                bsV, N_q, d = x.shape
                bs = bsV // num_views
                x_sa = x.reshape(bs, num_views * N_q, d)
                gate_sa = (
                    residual_gate.reshape(bs, num_views, 1, 1)[:, 0]
                    if residual_gate is not None else None
                )
                self_out = layer["self_attn"](x_sa)
                if gate_sa is not None:
                    self_out = self_out * gate_sa
                x = (x_sa + self_out).reshape(bsV, N_q, d)
            else:
                self_out = layer["self_attn"](x)
                if residual_gate is not None:
                    self_out = self_out * residual_gate
                x = x + self_out

            ffn_out = layer["ffn"](x)
            if residual_gate is not None:
                ffn_out = ffn_out * residual_gate
            x = x + ffn_out
        return x


class MemoryBank:
    """Eval-time per-episode accumulator.

    Caches PRE-extracted PaliGemma stage-1 tokens for the anchor and the
    K most-recent keyframes. Stage-1 cameras are fixed across the episode
    and the memory KV path is detached, so re-running PaliGemma on the
    anchor / history every step is wasted compute — we extract once and
    reuse.

    Two things are still kept as raw PCs for the anchor only:
      * anchor_pc / anchor_img_feat — used at STAGE 2 to render the anchor
        under the current step's zoom (loc=wpt, sca=st_sca). The zoom
        changes per step, so stage-2 anchor tokens cannot be cached.

    History does NOT need raw PCs for the forward pass: temporal memory
    is stage-1-only, and stage-1 cameras are fixed, so the cached
    stage-1 tokens are reused verbatim every step. However, raw PCs
    and img_feats ARE optionally stored for visualization so that the
    eval-time memory_grid.png can re-render each history frame through
    the same MVT renderer pipeline the model uses.

    NOT used for training: each batch element fetches its own anchor /
    history bundle from the GemBench dataset and the network re-extracts
    tokens fresh every step (necessary because batches are independent).
    """

    def __init__(self, k_temporal: int = 2, select: str = "sliding"):
        self.k = int(k_temporal)
        # ``select`` mirrors the dataset's ``memory_select``:
        #   "sliding"     -> legacy: every admitted keyframe gets a slot,
        #                    slot 0 = newest (chronological accumulator).
        #   "keyframe_gt" -> slot 0/1 pinned to the two most-recent EXECUTED
        #                    keyframes, slots 2..K-1 = subtask-boundary memory
        #                    keyframes admitted with a 2-step delay.
        self.select = str(select)
        self.reset()

    def reset(self):
        # Anchor (frame 0). Raw PC kept for stage-2 zoom re-rendering.
        self.anchor_pc = None              # list[Tensor (N_i, 3)] per sample
        self.anchor_img_feat = None        # list[Tensor (N_i, F)] per sample
        # Cached stage-1 PaliGemma tokens (extracted once, reused every step).
        self.anchor_mvt1_tokens = None     # Tensor (bs, vlm_dim, V, H_p, W_p)

        # ---- LEGACY sliding accumulator (select == "sliding") ----
        # CHRONOLOGICAL accumulator of admitted keyframes (append order:
        # bank[0]=oldest, bank[-1]=newest). ``ordered_slots`` reverses so
        # slot 0 = newest. ``self.k`` is the sliding-window cap.
        self.hist_mvt1_tokens = []         # list[ Tensor (bs, vlm_dim, V, H_p, W_p) ]
        # Raw world-frame PCs / img_feats for each history slot (viz only).
        self.hist_pc = []                  # list[ list[Tensor (N_i, 3)] ]
        self.hist_img_feat = []            # list[ list[Tensor (N_i, F)] ]

        # ---- keyframe_gt: recency window + delayed memory list ----
        # ``_recent`` is the 2-slot window of the most-recent EXECUTED
        # keyframes (chronological; _recent[-1] = newest). Each entry keeps
        # the detached stage-1 tokens, raw PC/feat (viz only) and the gate
        # decision (whether it should be admitted into the memory-keyframe
        # list once it scrolls out of the recency window).
        self._recent = []                  # list[ dict(tokens, pc, feat, gate) ]
        # Subtask-boundary keyframes that have LEFT the recency window
        # (frames <= t-3 at read time). Chronological (oldest first); only
        # gated frames. Capped at (k - 2) memory slots.
        self._mem_tokens = []              # list[ Tensor ]
        self._mem_pc = []                  # list[ list[Tensor] ]
        self._mem_img_feat = []            # list[ list[Tensor] ]

    def first_frame(self) -> bool:
        return self.anchor_pc is None

    def ordered_slots(self):
        """Filled temporal slots in tensor-slot order (slot 0 first).

        Returns a list (length <= k) of ``dict(tokens, pc, feat)``. For
        "keyframe_gt": slot 0/1 = the two most-recent executed keyframes
        (newest first), then the memory keyframes newest-first. For
        "sliding": the chronological accumulator reversed (slot 0 = newest).
        """
        slots = []
        if self.select == "keyframe_gt":
            for entry in reversed(self._recent):        # newest -> slot 0
                slots.append({"tokens": entry["tokens"],
                              "pc": entry["pc"], "feat": entry["feat"]})
            n_mem_slots = max(0, self.k - 2)
            for i in range(min(len(self._mem_tokens), n_mem_slots)):
                idx = len(self._mem_tokens) - 1 - i     # newest mem first
                slots.append({"tokens": self._mem_tokens[idx],
                              "pc": self._mem_pc[idx],
                              "feat": self._mem_img_feat[idx]})
            return slots[:self.k]
        # legacy sliding: chronological list, slot 0 = newest.
        n = len(self.hist_mvt1_tokens)
        for k in range(n):
            idx = n - 1 - k
            slots.append({"tokens": self.hist_mvt1_tokens[idx],
                          "pc": self.hist_pc[idx],
                          "feat": self.hist_img_feat[idx]})
        return slots

    def num_history(self) -> int:
        return len(self.ordered_slots())

    def set_anchor(self, pc, img_feat, mvt1_tokens):
        """Record the first-frame anchor at episode start. Called once
        per episode. ``mvt1_tokens`` is the post-PaliGemma stage-1 token
        grid for frame 0 (detached, bf16) — by construction equal to the
        very first ``act()``'s current-frame tokens, so the agent passes
        them in directly without an extra forward.
        """
        self.anchor_pc = pc
        self.anchor_img_feat = img_feat
        self.anchor_mvt1_tokens = mvt1_tokens

    def push(self, mvt1_tokens, gate: bool = True, pc_world=None, img_feat=None):
        """End-of-step push. ``mvt1_tokens`` is the DETACHED post-PaliGemma
        stage-1 token grid for the just-completed step.

        ``gate``: the discriminator decision (sigmoid(mem_logit) > threshold)
        = "is this a subtask-boundary memory keyframe". Defaults True so the
        legacy sliding-window callers keep working.

        ``pc_world`` / ``img_feat`` are optional post-bound raw PCs kept
        only for visualization (re-rendering the memory_grid). Pass None
        when visualization is disabled to save memory.

        select == "keyframe_gt":
          Every executed keyframe enters the 2-slot recency window
          regardless of ``gate``. The frame leaving the window (when a 3rd
          is pushed) is admitted into the memory-keyframe list ONLY if its
          gate was True. This 2-step delay keeps a keyframe from occupying
          a recency slot AND a memory slot at once (no duplication).

        select == "sliding":
          Chronological append of gated frames (legacy), capped at ``self.k``.
        """
        if self.select == "keyframe_gt":
            # Evict the frame leaving the 2-slot recency window into the
            # memory-keyframe list (only if it was a subtask boundary).
            if len(self._recent) >= 2:
                evicted = self._recent.pop(0)
                if evicted["gate"]:
                    self._mem_tokens.append(evicted["tokens"])
                    self._mem_pc.append(evicted["pc"])
                    self._mem_img_feat.append(evicted["feat"])
                    cap = max(0, self.k - 2)
                    if cap > 0 and len(self._mem_tokens) > cap:
                        self._mem_tokens = self._mem_tokens[-cap:]
                        self._mem_pc = self._mem_pc[-cap:]
                        self._mem_img_feat = self._mem_img_feat[-cap:]
            self._recent.append({
                "tokens": mvt1_tokens, "pc": pc_world,
                "feat": img_feat, "gate": bool(gate),
            })
            return

        if not gate:
            return
        self.hist_mvt1_tokens.append(mvt1_tokens)
        self.hist_pc.append(pc_world)
        self.hist_img_feat.append(img_feat)
        # Safety cap only (M sized to cover all tasks; overflow is rare):
        # drop the OLDEST to mirror the dataset's sel[-M:] (keep most-recent M).
        if self.k is not None and self.k > 0 and len(self.hist_mvt1_tokens) > self.k:
            self.hist_mvt1_tokens = self.hist_mvt1_tokens[-self.k:]
            self.hist_pc = self.hist_pc[-self.k:]
            self.hist_img_feat = self.hist_img_feat[-self.k:]


def stack_hist_tokens_from_bank(bank: "MemoryBank", bs_act: int, K: int, device):
    """Stack cached history into ``(bs, K, vlm_dim, V, H_p, W_p)`` tensors.

    Tensor slot ``k`` follows ``bank.ordered_slots()`` (slot 0 first):
      * "sliding"     -> slot k = k-th most recent admitted keyframe.
      * "keyframe_gt" -> slot 0/1 = the two most-recent executed keyframes,
        slots 2..K-1 = subtask-boundary memory keyframes (newest first).
    """
    slots = bank.ordered_slots()
    n_hist = len(slots)
    hist_mask_t = torch.zeros(bs_act, K, dtype=torch.bool, device=device)
    hist_tokens_t = None
    if n_hist > 0:
        tpl = slots[0]["tokens"]
        hist_tokens_t = tpl.new_zeros((bs_act, K, *tpl.shape[1:]))
        for k in range(min(n_hist, K)):
            hist_tokens_t[:, k] = slots[k]["tokens"]
            hist_mask_t[:, k] = True
    return hist_tokens_t, hist_mask_t


def bank_idx_for_hist_slot(n_hist: int, slot: int) -> int:
    """Map tensor slot (0=newest) to ``MemoryBank`` list index (0=oldest)."""
    assert 0 <= slot < n_hist
    return n_hist - 1 - slot


# Diagnostic helpers for tracking memory-branch learning progress.
def _norms_for_block(block: "MemoryBlock", tag: str) -> dict:
    """Extract per-layer + aggregate L2 weight norms for one MemoryBlock.

    Emits, for each layer ``i`` in [0, num_layers):
      * ``{tag}/layer{i}/cross_to_out``       — cross-attn output projection
      * ``{tag}/layer{i}/self_attn_to_out``   — self-attn output projection
      * ``{tag}/layer{i}/ffn_out``            — FFN's last Linear

    Plus a block-level aggregate:
      * ``{tag}/total_to_out`` — L2 norm of the three to_out tensors stacked
        across every layer. With default (non-zero) init this starts at a
        non-zero value and should drift as training proceeds; a flat curve
        means no gradient is flowing through the memory branch.

    For temporal blocks (``block.kind == "temporal"``) one extra scalar is
    reported so you can confirm slot conditioning is also moving:
      * ``{tag}/slot_embed``
    """
    out = {}
    total_sq = 0.0
    for li, layer in enumerate(block.layers):
        # ``.detach().float()`` because memory blocks live in fp32 by
        # construction, but downstream casts (bf16 backbone) could in
        # principle migrate them; .float() makes the norm dtype-agnostic.
        n_cross = float(layer["cross"].fn.to_out.weight.detach().float().norm())
        n_self = float(layer["self_attn"].fn.to_out.weight.detach().float().norm())
        n_ffn = float(layer["ffn"].fn.net[-1].weight.detach().float().norm())
        out[f"{tag}/layer{li}/cross_to_out"] = n_cross
        out[f"{tag}/layer{li}/self_attn_to_out"] = n_self
        out[f"{tag}/layer{li}/ffn_out"] = n_ffn
        total_sq += n_cross ** 2 + n_self ** 2 + n_ffn ** 2
    out[f"{tag}/total_to_out"] = total_sq ** 0.5

    if block.kind == "temporal":
        out[f"{tag}/slot_embed"] = float(
            block.slot_embed.weight.detach().float().norm()
        )
    return out


@torch.no_grad()
def memory_param_norms(model, prefix: str = "mem_norm") -> dict:
    """Return a flat dict of L2 weight norms for all MemoryBlock parameters
    we want to track over training time.

    The residual-exit projections (``cross.to_out``, ``self_attn.to_out``,
    ``ffn`` last linear) use the default initialisation of their submodules
    (see ``MemoryBlock.__init__``), so their norms start at non-zero values
    and should drift as training proceeds. A flat curve means the memory
    branch is being skipped (no gradient flowing through it) — either
    because the data has no anchor/history, the loss never backprops
    through cross-attn, or the memory params were frozen.

    ``model`` may be (auto-unwrapped in this order):
      * A DDP-wrapped ``MVT`` wrapper        (training time)
      * A bare ``MVT`` wrapper (``self.mvt1`` present)
      * A bare ``MVTSingle`` (memory attrs directly on it)
      * A bare ``MemoryBlock``                (unit-test path)

    Three logical block tags are emitted (only the ones the model actually
    has are reported — disabling ``spatial_at_mvt2`` etc. silently drops
    that tag):
      * ``spatial_s1``  — anchor cross-attn injected at zoom-stage 1
      * ``spatial_s2``  — anchor cross-attn injected at zoom-stage 2
      * ``temporal_s1`` — K-slot sliding-window cross-attn (stage 1 only)
    """
    # DDP -> MVT wrapper -> MVTSingle. Each unwrap is a no-op if the
    # attribute is absent, so passing any of the three shapes Just Works.
    inner = getattr(model, "module", model)
    inner = getattr(inner, "mvt1", inner)

    # Direct MemoryBlock case (unit-test path).
    if isinstance(inner, MemoryBlock):
        return _norms_for_block(inner, f"{prefix}/{inner.kind}")

    out = {}
    for attr, tag in (
        ("mem_spatial_s1",  "spatial_s1"),
        ("mem_spatial_s2",  "spatial_s2"),
        ("mem_temporal_s1", "temporal_s1"),
    ):
        block = getattr(inner, attr, None)
        if block is not None:
            out.update(_norms_for_block(block, f"{prefix}/{tag}"))
    return out
