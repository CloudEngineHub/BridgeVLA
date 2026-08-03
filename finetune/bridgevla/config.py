# Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.agent = "our"
_C.tasks = "insert_onto_square_peg,open_drawer,place_wine_at_rack_location,light_bulb_in"
_C.exp_id = "def"
# bs per device, effective bs is scaled by num device
_C.bs = 4
_C.epochs = 100
# number of dataloader workers, >= 0
_C.num_workers = 0
# 'transition_uniform' or 'task_uniform'. Only consumed by Colosseum's YARR-replay-buffer path; every other
# bench ignores it and uses the ``task_sampling`` block below instead.
_C.sample_distribution_mode = 'transition_uniform'

# --- Multi-task sampling balance (RLBench / RMBench_vla / real) ---
# Both flatten episodes into a per-(task, ep, step) transition index drawn by a uniform DistributedSampler,
# so each task is sampled proportional to its TOTAL keyframe count and long-horizon tasks dominate (~20x more
# updates/epoch for blocks_ranking_try vs observe_and_pickup in RMBench_vla). Temperature sampling draws
# task t with probability ∝ n_t**alpha, i.e. each transition gets weight ∝ n_t**(alpha-1):
#   alpha = 1.0 -> transition_uniform (original; long tasks dominate)
#   alpha = 0.0 -> task_uniform
#   alpha = 0.5 -> tempered middle ground (recommended)
# Other benches ignore this block.
_C.task_sampling = CN()
_C.task_sampling.mode = "transition_uniform"   # "transition_uniform" | "temperature"
_C.task_sampling.alpha = 0.5                    # only used when mode == "temperature"
# Epoch length as a MULTIPLE of the dataset size. 1.0 (default) = one classic pass. >1 lengthens the epoch,
# useful for small datasets where a 1x epoch is only a few hundred steps and per-epoch bookkeeping
# (checkpoints, viz, SwanLab rows, freeze-stage switching) would otherwise fire every few seconds.
# Total training length is unchanged if the epoch-denominated knobs (epochs / save_every_n_epochs /
# freeze_epochs / viz_every_n_epochs) are scaled down by the same factor. warmup_steps counts STEPS and must
# NOT be rescaled. Consumed by real/train.py only.
_C.task_sampling.epoch_size_mult = 1.0
# Level at which the temperature acts (real/train.py only).
#   "task"     -> flat over task_group dirs: p_t ∝ n_t**alpha (the RLBench recipe).
#   "category" -> two-level: p_c ∝ n_c**alpha over coarse SKILL categories, then uniform across the language
#                 variants inside a category. Use when task dirs encode skill x object-variant and the variant
#                 count differs per skill (real: 19 dirs -> 5 skills), so a flat balance would reward whichever
#                 skill happened to get more nouns recorded. Needs category_transition_counts()/category_to_tasks().
_C.task_sampling.group_mode = "task"

_C.train_iter = 16 * 10000
_C.use_scheduler = True
_C.swanlab_project = "bridgevla_plus"
# Base name for the SwanLab run; the final run/folder name is f"{swanlab_run}_{MM_DD_HH_MM}".
_C.swanlab_run = "bridgevla"
# Checkpoint save period in epochs; 0 disables periodic saving. Epoch 0 is skipped, the final epoch always saved.
_C.save_every_n_epochs = 10
# False (default): checkpoints store only model weights (much smaller); resume falls back to a fresh
# optimizer + global_step=0. True: also store optimizer state + global_step + freeze_epochs for an exact
# resume. Consumed by RMBench_vla/train.py and real/train.py.
_C.save_optimizer_state = False
# root directory for train logs; final path is
#   f"{log_dir}/train/{run_name}/"
_C.log_dir = "data/bridgevla_data/logs"  # fallback; launchers pass --log_dir explicitly
# arguments present in both peract and rvt
# some of them donot support every possible combination in peract
_C.peract = CN()
_C.peract.lambda_weight_l2 = 1e-6
# lr should be thought on per sample basis
# effective lr is multiplied by bs * num_devices
_C.peract.lr = 2.5e-5
_C.peract.optimizer_type =  "adam" # "lamb"
_C.peract.weight_decay = 0.01
_C.peract.betas = [0.9, 0.95]
_C.peract.add_rgc_loss = True
_C.peract.num_rotation_classes = 72
_C.peract.transform_augmentation = True
_C.peract.transform_augmentation_xyz = [0.1, 0.1, 0.1]
_C.peract.transform_augmentation_rpy = [0.0, 0.0, 20.0]

# arguments present in only rvt and not peract
# Two-stage training
_C.freeze_epochs = 5       # Stage 1: freeze backbones for this many epochs
_C.warmup_steps = 2000     # Linear warmup steps per stage

# Per-task-episode heatmap visualization cadence (GemBench / RLBench / real). Every N epochs rank 0 picks
# one random episode per task, runs eval mode over its steps, and dumps pred/GT overlays under
# {log_dir}/viz/epoch_{epoch:04d}/{task}/{episode}/step_{k:03d}/. 0 disables it.
_C.viz_every_n_epochs = 20

# Per-task viz whitelist (GemBench / real): None -> all tasks (default); [] -> disable; a list -> only those.
# Names must match the directory names under data/keysteps_bbox/seed0/; unknown names are ignored.
_C.viz_tasks = None

# Whether to run the per-task viz pass at the very start of training (epoch 0), capturing a pre-training
# baseline. Periodic and final-epoch viz are unaffected by this flag.
_C.initial_viz = True

_C.rvt = CN()
_C.rvt.gt_hm_sigma = 1.5
_C.rvt.img_aug = 0.1
_C.rvt.place_with_mean = True
_C.rvt.move_pc_in_bound = True
# Kendall-Gal per-view uncertainty weighting (deferred ablation). Default False = uniform 1/N averaging;
# the True branch raises NotImplementedError on the current PaliGemma-direct path.
_C.rvt.use_view_logvar = False

# The item-4 ablation (modified focal loss + dual-head post-processing) is controlled by
# ``mvt_cfg.use_modified_focal_loss``; the agent picks it up from the MVT module attribute.

# arguments present in peract official
_C.peract_official = CN()
_C.peract_official.cfg_path = "configs/peract_official_config.yaml"

# --- PaliGemma activation (gradient) checkpointing: mirror of ``mvt_cfg.gradient_checkpointing``, settable
# from the bench yaml and propagated onto mvt_cfg. Default True (use_reentrant=False, DDP-safe). ---
_C.gradient_checkpointing = True

# Mirror of ``mvt_cfg.rotate_top_ccw90``; RMBench sets True to roll the tri-view top camera CCW 90°.
_C.rotate_top_ccw90 = False
# Mirror of ``mvt_cfg.renderer_img_sizes_w`` (smaller = tighter mvt1/mvt2 FOV).
_C.renderer_img_sizes_w = 2.0
# Mirror of ``mvt_cfg.splat_radius`` / ``splat_radius_mvt2`` (mvt1 / mvt2).
_C.splat_radius = 0.012
_C.splat_radius_mvt2 = 0.012
# Mirror of ``mvt_cfg.st_wpt_loc_aug`` (stage-2 zoom-center jitter).
_C.st_wpt_loc_aug = 0.05
# Mirror of ``mvt_cfg.img_aug_2``. Memory-enabled configs should set 0.0 so the current-frame stage-2 render
# noise stays aligned with the deterministic anchor/history renders.
_C.img_aug_2 = 0.0

# --- Dual-arm (RMBench): mirror of ``mvt_cfg.num_arms``, set from the bench yaml and propagated onto
# mvt_cfg before the network is built. The default keeps single-arm benches unchanged. ---
_C.num_arms = 1

# --- Per-step ignore-collisions supervision ---
# Passed to ``RVTAgent(predict_collision=...)``, gating the label read and the 2-way collision CE. It does
# NOT change the head width — the feat head always emits grip(2)+collision(2) — so checkpoints stay
# loadable either way.
#   True  — RLBench / Colosseum / real: a genuine per-keyframe label, routed into the path planner at eval.
#   False — GemBench / memoryBench / RMBench: no usable label, so the key is never emitted and eval ignores it.
_C.predict_collision = True

# --- Rotation head representation ---
# "euler_disc" (default): the original discrete-Euler classification head (3 * num_rotation_classes logits
#     + per-axis CE, gimble_fix).
# "6d": continuous 6D rotation regression (Zhou et al., CVPR 2019). train.py maps it to mvt_cfg.rot_ver = 2
#     and shrinks feat_dim to 6 + grip(2) + collision(2) = 10, fully bypassing discretization / gimble_fix.
_C.rotation_representation = "euler_disc"

# Mirror of ``mvt_cfg.feat_from_stage1``. False (default): rot/grip/collision features come from stage 2
# (fine zoom). True (RMBench): from stage 1 (global max-pool + waypoint local patch).
_C.feat_from_stage1 = False

# --- Episodic memory (spatial anchor + temporal episodic) ---
# Mirror of ``mvt_cfg.memory``, so all settings can live in one bench yaml; train.py propagates this block
# onto mvt_cfg.memory before the network is built. Defaults match the mvt-side defaults.
_C.memory = CN()
_C.memory.enabled = False
_C.memory.k_temporal = 4
_C.memory.spatial_at_mvt1 = True
_C.memory.spatial_at_mvt2 = True
_C.memory.temporal_at_mvt1 = True
# Memory ablation master switches (mirror of mvt_cfg.memory); effective = fine_flag AND switch.
#   temporal_memory: stage-1 temporal block (2 recency + history keyframes) + the stage-1 spatial anchor.
#   spatial_memory:  stage-2 per-view spatial anchor.
_C.memory.temporal_memory = True
_C.memory.spatial_memory = True
_C.memory.share_spatial_across_stages = False
_C.memory.heads = 8
_C.memory.dim_head = 128
_C.memory.num_layers = 2
_C.memory.ffn_mult = 2
_C.memory.use_fast_attn = False
# Temporal-memory frame-selection policy ("keyframe_gt" | "sliding"); slot 0 is always the most recent keyframe.
_C.memory.select = "keyframe_gt"
# Keyframe discriminator (mirror of mvt_cfg.memory.discriminator; propagated onto mvt_cfg).
_C.memory.discriminator = CN()
_C.memory.discriminator.enabled = False
_C.memory.discriminator.input = "current"
_C.memory.discriminator.heads = 8
_C.memory.discriminator.hidden = 512
_C.memory.discriminator.loss_weight = 1.0
_C.memory.discriminator.pos_weight = 5.5
_C.memory.discriminator.threshold = 0.5
# True (default): gradient flows through the memory KV tokens back into PaliGemma (anchor / history
# forwards build a graph, KV not detached). False restores the legacy no_grad + detached path.
_C.memory.grad_through_tokens = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()


def merge_from_file_drop_unknown(cfg, path, where=""):
    """``cfg.merge_from_file(path)`` that tolerates keys this schema dropped.

    A checkpoint's exp_cfg.yaml is a snapshot of the schema at training time, so
    older ckpts carry blocks that later cleanups removed (e.g. ``pretrain_mix``,
    deleted together with GemBench's pretrain-grounding mix). yacs' merge raises
    KeyError on any such key, which would make every pre-cleanup ckpt
    un-evaluable. Unknown keys are inert by construction — nothing in the
    current code reads them — so they are dropped, but loudly: a silently
    ignored key must never look like an applied setting.

    Returns the list of dropped dotted key paths.
    """
    import yaml

    with open(path, "r") as fp:
        loaded = yaml.safe_load(fp) or {}

    dropped = []

    def prune(node, schema, prefix):
        kept = {}
        for key, value in node.items():
            if key not in schema:
                dropped.append(prefix + str(key))
                continue
            if isinstance(value, dict) and isinstance(schema[key], CN):
                kept[key] = prune(value, schema[key], f"{prefix}{key}.")
            else:
                kept[key] = value
        return kept

    cfg.merge_from_other_cfg(CN(prune(loaded, cfg, "")))
    if dropped:
        tag = f" [{where}]" if where else ""
        print(f"[config]{tag} dropped {len(dropped)} key(s) absent from the "
              f"current schema while loading {path}: {', '.join(dropped)}")
    return dropped
