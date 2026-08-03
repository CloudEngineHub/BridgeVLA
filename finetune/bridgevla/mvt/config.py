# Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/mvt/config.py
from yacs.config import CfgNode as CN
_C = CN()
_C.depth = 8
_C.img_size = 220
_C.img_feat_dim = 3
_C.feat_dim = (72 * 3) + 2 + 2
_C.im_channels = 64
_C.activation = "lrelu"
_C.decoder_dropout = 0.0
_C.img_patch_size = 11
_C.final_dim = 64
_C.self_cross_ver = 1
_C.add_corr = True
_C.norm_corr = False
_C.add_pixel_loc = True
_C.add_depth = True
_C.rend_three_views = False
# True: three mutually perpendicular oblique orthographic cameras looking down (the eye makes 54.7356°
# with +Z, azimuths 0/120/240). Implies 3 views; mutually exclusive with rend_three_views' axis-aligned top/front/right.
_C.rend_oblique_views = False
_C.use_point_renderer = False
_C.pe_fix = True
_C.feat_ver = 0
_C.wpt_img_aug = 0.01
_C.inp_pre_pro = True
_C.inp_pre_con = True
_C.cvx_up = False
_C.xops = False
_C.rot_ver = 0
_C.num_rot = 72
_C.stage_two = True

# Rotation / gripper / collision head FEATURE SOURCE.
# False (default): from stage 2 (mvt2, zoomed-in fine render) — the original BridgeVLA behavior, reading
#   high-resolution local features around the zoomed waypoint.
# True (RMBench option): from stage 1 (global max-pool + waypoint local patch), leaving stage 2 to emit only
#   the translation heatmap. Translation still uses both stages' heatmaps.
_C.feat_from_stage1 = False

# ---- Dual-arm support (RMBench) ----
# num_arms == 1 (default): single-arm, byte-identical to the original BridgeVLA (flat head names, one
# heatmap, one (trans, rot, grip) prediction). num_arms > 1 (RMBench = 2): the PaliGemma trunk + memory
# blocks are SHARED across arms (run once at mvt1, once per arm at mvt2) and only the heads branch, into
# per-arm ``up0_arms[i]`` and ``feat_fc_*_arms[i]``.
_C.num_arms = 1
# Documentation mirror of ``exp_cfg.predict_collision`` only — MVT / MVTSingle store it but never read it,
# and the feat head always emits grip(2)+coll(2) regardless. The flag that acts is the one passed to
# ``RVTAgent(predict_collision=...)``. Kept here so the yamls and the built network agree and a checkpoint
# records which mode it was trained in.
_C.predict_collision = True
_C.st_sca = 4
_C.st_wpt_loc_aug = 0.05
_C.st_wpt_loc_inp_no_noise = False
_C.img_aug_2 = 0.0
_C.flip_top_up = False
# Roll the tri-view renderer's top camera up-vector CCW 90° about +Z, fixing top-view orientation mismatches
# without touching the front/right cameras. RMBench enables this.
_C.rotate_top_ccw90 = False
# Orthographic renderer world extent [W, H] in cube coords (2.0 covers [-1,1]^3); smaller zooms mvt1/mvt2 in.
_C.renderer_img_sizes_w = 2.0

# Point-cloud splat radius (WORLD units) for the mvt1 tri-view renderer; splat_radius_mvt2 defaults to 0.012
# so stage 2 can stay coarser when mvt1 is tuned finer. Pixel kernel = ceil(focal_px * radius).
_C.splat_radius = 0.012
_C.splat_radius_mvt2 = 0.012

# Modified focal loss + dual ConvexUpSample heads (item 4 ablation). False (default) = the original single
# ``up0`` head + softmax+CE; True = up_action / up_grounding heads with per-pixel sigmoid focal loss.
_C.use_modified_focal_loss = False

# Kendall-Gal per-view uncertainty weighting (deferred ablation). False (default) = uniform 1/N averaging;
# the True branch raises NotImplementedError until the q_weight pipeline is rebuilt.
_C.use_view_logvar = False

# ---- PaliGemma activation checkpointing ----
# True (default): HF gradient checkpointing on the backbone (vision tower + decoder layers) with
# ``use_reentrant=False``. Trades ~1 extra forward per backward for a large drop in activation memory —
# essential with ``memory.grad_through_tokens=True``, which adds anchor + K history graph-building forwards.
# use_reentrant=False is required for DDP ``find_unused_parameters=True`` and tolerates inputs without
# requires_grad (the fully-frozen Stage 1). ``model.config.use_cache`` is forced False at enable time.
_C.gradient_checkpointing = True

# ---- Episodic memory (spatial anchor + temporal episodic) ----
# Disabled by default so runs without the memory module stay bit-for-bit unchanged. When enabled:
#   spatial_at_mvt1/2: cross-attend the current PaliGemma tokens against the episode's frame-0 ("anchor")
#     tokens. At stage 2 the anchor PC is re-rendered with the same trans_pc as the current frame, so both
#     share zoom-local camera coordinates.
#   temporal_at_mvt1: cross-attend against the K_temporal most recent keyframes, with a per-slot learned
#     embedding added to the KV before to_kv. Purely visual — historical-action PE was removed. Stage 2
#     deliberately has no temporal block.
# Anchor + history forwards either backprop into PaliGemma (default) or run under no_grad with detached KVs.
_C.memory = CN()
_C.memory.enabled = False
_C.memory.k_temporal = 4
_C.memory.spatial_at_mvt1 = True
_C.memory.spatial_at_mvt2 = True
_C.memory.temporal_at_mvt1 = True
# --- Memory ablation master switches (TPAMI; per-stage memory groups) ---
# These gate WHOLE memory groups on top of the fine-grained spatial_at_mvt*/temporal_at_mvt1 flags
# (effective = fine_flag AND switch). The stage-1 / stage-2 zoom pipeline is unchanged either way; only the
# episodic-memory cross-attention injections are toggled.
#   temporal_memory: STAGE-1 memory = the temporal block (``mem_temporal_s1``) AND the stage-1 spatial
#       anchor (``mem_spatial_s1``). false -> no stage-1 memory at all.
#   spatial_memory:  STAGE-2 per-view spatial anchor (``mem_spatial_s2``).
# Default True (full memory).
_C.memory.temporal_memory = True
_C.memory.spatial_memory = True
# Reserved for ablation: True makes mvt1 and mvt2 share one spatial block. Default False (independent params).
_C.memory.share_spatial_across_stages = False
# MemoryBlock attention shape; block dim follows PaliGemma's image-token hidden size (vlm_dim=2048). With
# heads=8 and dim_head=128 the inner Q/K/V dim is 1024. ``num_layers`` stacks (cross + self + ffn) layers,
# all reusing the same KV.
_C.memory.heads = 8
_C.memory.dim_head = 128
_C.memory.num_layers = 2
_C.memory.ffn_mult = 2
_C.memory.use_fast_attn = False
# True (default): anchor / history PaliGemma forwards run with grad enabled and their KV tokens are NOT
# detached before the memory cross-attention, so the backbone gets a learning signal from the memory branch
# — at the cost of an extra backward through PaliGemma per memory entry. False restores the legacy no_grad +
# detached path. Eval is unaffected (bank tokens are already detached and inference is under no_grad).
_C.memory.grad_through_tokens = True
# Temporal-memory frame-selection policy: "keyframe_gt" = variable-length memory of GT subtask-boundary
# keyframes (k_temporal is the cap M); "sliding" = legacy K most-recent. Slot 0 is the newest either way.
_C.memory.select = "keyframe_gt"
# Keyframe discriminator: predicts whether the current frame is a subtask-boundary keyframe worth admitting
# into temporal memory. Input = the current frame's post-memory stage-1 tokens; BCE on mem_label; gates the
# eval MemoryBank push. Disabled by default.
_C.memory.discriminator = CN()
_C.memory.discriminator.enabled = False
_C.memory.discriminator.input = "current"   # current = post-memory current tokens
_C.memory.discriminator.heads = 8
_C.memory.discriminator.hidden = 512
_C.memory.discriminator.loss_weight = 1.0    # lambda on the BCE term
_C.memory.discriminator.pos_weight = 5.5     # BCEWithLogits pos_weight (~neg/pos)
_C.memory.discriminator.threshold = 0.5      # eval sigmoid gate


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
