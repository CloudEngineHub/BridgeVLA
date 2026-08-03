# MemoryBench finetuning + evaluation for BridgeVLA

End-to-end scaffolding to train and evaluate BridgeVLA on the 3 MemoryBench
tasks (`put_block_back`, `rearrange_block`, `reopen_drawer`). Mirrors the
GemBench layout 1:1 and reuses the same conda env, simulator and agent — the
only substantive deltas are:

- **Dataset**: GemBench reads pre-built LMDB samples; here we read the
  released RLBench-format MemoryBench data and run `keypoint_discovery()` to
  extract one sample **per (k → k+1) keyframe transition**, no recurring
  keypoints, no random non-keyframe sampling. See `memorybench_dataset.py`
  for details and the on-disk cache format.
- **Tasks**: 3 task `.py` + `.ttm` files that no upstream RLBench carries.
  They are **pinned in-repo** at `finetune/bridgevla/libs/rlbench_patches/tasks/`
  (the exact versions every released checkpoint was trained and evaluated with),
  and `scripts/install_memorybench_tasks.sh` copies them into the RLBench source
  stack on `PYTHONPATH` — enough for `task.set_variation()` / `task.reset()`.
  The upstream data release's `data/files/` has since evolved and may differ;
  point `SRC_DIR=` at it only if you deliberately want that version.
- **Episode horizon**: client uses `MAX_STEPS=50` (vs GemBench's 25) because
  MemoryBench tasks routinely produce 10+ keyframes (lift, navigate, press,
  navigate back, lower, release).
- **Model logic aligned with RMBench** (the deltas vs the old GemBench-style
  memoryBench setup):
  - **Rotation head = 6D regression** (`rotation_representation: "6d"` →
    `rot_ver=2`, `feat_dim=10`): the gripper rotation is regressed as a 6D
    vector (Zhou et al., CVPR 2019) → Gram–Schmidt → SO(3), supervised by a
    Frobenius loss to the GT rotation matrix — replacing the discrete /
    autoregressive Euler head.
  - **Rot/grip/collision feature source = stage 1** (`feat_from_stage1: true`):
    the rotation/gripper/collision head reads coarse stage-1 features; stage 2
    then emits only the fine translation heatmap (translation precision is
    unchanged — still stage1+stage2 heatmaps).
  - **Episodic memory = `keyframe_gt`** (`memory.select: keyframe_gt`,
    `k_temporal: 2`): the temporal memory is the frame-0 anchor + the two
    most-recent EXECUTED keyframes ("near neighbour two frames"), matching
    RMBench's recency slots (slot 0/1 = t-1, t-2).
  - **Keyframe discriminator disabled** (`memory.discriminator.enabled:
    false`): RMBench uses a discriminator to admit GT subtask-boundary
    keyframes into the extra memory slots, but the real MemoryBench data has no
    boundary label, so the label defaults to false and the admission gate is
    always false — nothing is ever admitted and the memory stays exactly anchor
    + the two recency frames (`k_temporal=2` → no extra memory slots).
  - **Unchanged**: point-cloud granularity / renderer hyper-params, input
    cameras + `IMAGE_SIZE`, single-arm + collision prediction, and the
    post-inference 9-D action handed to the RLBench planner.

## Conda env

**Reuse `bridgevla_plus_gembench`** — same simulator, PyRep, RLBench, peract
helpers, BridgeVLA agent. No second env needed. The 3 MemoryBench tasks just
need to be copied into the RLBench source stack (one-shot via
`scripts/install_memorybench_tasks.sh`, called automatically by `train.sh` /
`run_*.sh`).

## Layout

```
finetune/memoryBench/
├── memorybench_dataset.py     # GemBench-style keyframe sampling, RLBench raw input
├── train.py                    # mirror of GemBench/train.py with this dataset
├── train.sh                    # launcher (torchrun, env vars, swanlab)
├── server.py                   # Flask wrapper around MyActioner
├── client.py                   # rollout loop (RLBenchEnv + Mover)
├── actioner.py                 # MyActioner: hosts the BridgeVLA agent
├── run_server.sh
├── run_client.sh
├── summarize_eval.py           # parse per-task jsonl logs into a table
├── configs/
│   └── memorybench_config.yaml
├── assets/
│   ├── taskvars.json           # 9 taskvars across 3 tasks
│   └── taskvars_instructions.json
├── utils/
│   ├── __init__.py
│   └── peract_utils_memorybench.py
└── scripts/
    ├── install_memorybench_tasks.sh   # idempotent RLBench task install (from the in-repo pin)
    ├── unzip_memorybench_data.sh       # untar the data zips in-place
    ├── reorg_data_for_eval.py          # build per-variation symlinks
    └── viz_episode_sampling.py         # plot which keyframes an episode samples
```

## Data preparation

Released MemoryBench data lives at:

```
data/bridgevla_data/memorybench/
├── data/
│   ├── files/                  # 3× .py + 3× .ttm as shipped upstream (NOT what we install —
│   │                           #   install_memorybench_tasks.sh uses the in-repo pin instead)
│   ├── train/<task>.zip        # 100 episodes / task
│   └── test/<task>.zip
```

Step 1 — extract the train + test zips (only run once):

```bash
bash scripts/unzip_memorybench_data.sh
```

Each zip contains `<task>/all_variations/episodes/episode<i>/` with
`low_dim_obs.pkl`, `variation_descriptions.pkl`, `variation_number.pkl` and
the per-camera RGB / depth / mask frames — i.e. the standard sam2act layout.

Step 2 (optional, only needed if you want `task.reset_to_demo()` style
evaluation) — build per-variation symlinks:

```bash
python scripts/reorg_data_for_eval.py \
  --src data/bridgevla_data/memorybench/data/test \
  --dst data/bridgevla_data/memorybench/eval_layout/test
```

Step 3 — install the 3 pinned MemoryBench task definitions into the RLBench
source stack (automatic from `train.sh` / `run_*.sh`, but you can run it manually):

```bash
bash scripts/install_memorybench_tasks.sh
```

## Sampling: how MemoryBench data becomes BridgeVLA training samples

`MemoryBench_Dataset` builds a small per-episode npz cache on first run:

1. Open `low_dim_obs.pkl` via `peract_colab.get_stored_demo` (this requires
   PyRep/CoppeliaSim importable — `train.sh` sets all the env vars).
2. Call `keypoint_discovery(demo)` (peract heuristic — gripper-state changes
   + zero-velocity stops). For `put_block_back` ep98 this yields ~12
   keypoints out of 355 frames.
3. For each keypoint `k`, resize the per-camera RGB+PCD to `IMAGE_SIZE=128`
   (a no-op — MemoryBench RLBench data is saved natively at 128) and store.
   Frame 0's obs is stored separately as the **anchor** view for episodic
   memory.
4. Cache to `data/_keyframe_cache/size128_v3/<task>/episode<i>.npz` plus a tiny
   `.meta` json with keyframe indices and language goal. The memory layout
   (`keyframe_gt` vs sliding) and `mem_label` are computed on the fly in
   `__getitem__`, so switching the memory policy does NOT invalidate this cache.

`__getitem__(idx)` then resolves to `(task, episode, step_idx)` and emits the
GemBench-compatible sample shape:

```python
{
  "front":          {"rgb": (3, 128, 128) uint8 → -1..1, "pcd": (3, 128, 128) float32},
  "left_shoulder":  ...,
  "right_shoulder": ...,
  "wrist":          ...,
  "gripper_pose":   (8,)   pos(3) + xyzw(4) + grip(1)   # action target
  "low_dim_state":  (9,)   gripper_pose + time_feat
  "ignore_collisions": float
  "lang_goal":      str
  "tasks":          str
  # if memory.enabled=True:
  "anchor_<cam>":   {"rgb": ..., "pcd": ...}            # frame 0
  "anchor_mask":    bool                                # False at step 0
  "hist{k}_<cam>":  {"rgb": ..., "pcd": ...}            # for k in [0..K-1]
  "hist_mask":      (K,) bool
  "mem_label":      float32                             # always 0 (no boundary label)
  "step_idx":       int64
}
```

With `memory.select: keyframe_gt` (the default), the history slots follow
RMBench's layout: slot 0 = previous executed keyframe (t-1), slot 1 = the one
before (t-2), and slots 2..K-1 = discriminator-admitted GT subtask-boundary
keyframes — which stay empty here (`mem_label` is all-zero), so with
`k_temporal=2` the memory is exactly the anchor + the two recency frames. The
legacy `sliding` policy (K most-recent keyframes) is still available.

This matches `bridgevla_agent.update_gembench` and
`_build_memory_inputs_from_replay` exactly so no agent code changes are
required.

### Why GemBench-style sampling (one sample per keyframe transition)?

- RLBench / sam2act `fill_replay()` iterates every `demo_augmentation_every_n
  = 10` non-keyframe and pairs it with the next keyframe → a single keyframe
  appears as the action target for many neighbouring frames. This biases
  training toward redundant samples.
- GemBench emits exactly one sample per (k → k+1), with the obs at keyframe
  k and the action at keyframe k+1. Cleaner curriculum, smaller dataset,
  stronger gradients per sample.

For 3 tasks × 100 episodes × ~10 keyframes each, the total dataset is a few
thousand samples — well-suited to short fine-tunes from a GemBench-pretrained
checkpoint.

## Training

```bash
cd finetune/memoryBench
WORLD_SIZE=1 RESOURCE_GPU=2 RANK=0 MASTER_ADDR=localhost MASTER_PORT=29622 \
  bash train.sh
```

Pass `--no-pretrain` to cold-start, or set `PRETRAIN_PATH=/path/to.pth` to
warm-start from a different checkpoint. By default it warm-starts from
BridgeVLA's released grounding pre-training weights
(`${BRIDGEVLA_CKPT_ROOT}/pretrain`, an HF sharded-safetensors directory); if you
ran `pretrain/pretrain.sh` yourself, point `PRETRAIN_PATH` at
`<run>/pretrain_epoch_<N>.pth` instead — both layouts load (see the root
README §3).

Memory ablation switches (same convention as RLBench / GemBench / RMBench_vla):

```bash
bash train.sh --temporal_memory false   # ablate memory 1 (stage-1 temporal)
bash train.sh --spatial_memory false    # ablate memory 2 (stage-2 anchor)
bash train.sh --temporal_memory false --spatial_memory false  # both off
```

CLI values override the YAML (`memory.temporal_memory` / `memory.spatial_memory`
in `configs/memorybench_config.yaml`); ablated runs automatically get a
`_no_temporal_mem` / `_no_spatial_mem` / `_no_mem` suffix on the swanlab run
name and run directory, and the switches are baked into the run's
`mvt_cfg.yaml` for the eval-time guard below.

Stage-1 / Stage-2 freeze schedule, optimizer rebuild across DDP, and
swanlab logging follow GemBench's `train.sh` byte-for-byte. The Stage-1 epoch
count is `freeze_epochs=20` by default (see `configs/memorybench_config.yaml`).

## Evaluation

Two terminals.

Terminal A — host the agent:

```bash
bash run_server.sh <model_epoch> <model_folder>
# e.g. bash run_server.sh 320 data/bridgevla_ckpt/bridgevla_plus/memorybench
```

Terminal B — roll out for every taskvar in `assets/taskvars.json`:

```bash
# Mode 1 (simplest): random task.reset() each episode.
bash run_client.sh <model_folder> <model_epoch>

# Mode 2: deterministic reset_to_demo from per-variation symlinks built by
# reorg_data_for_eval.py.
DEMO_DIR=/.../memorybench/eval_layout/test bash run_client.sh <model_folder> <model_epoch>
```

Evaluating a memory-ablated checkpoint: declare the expected switches on BOTH
sides — they must match the loaded model's `mvt_cfg.yaml` or the run aborts at
startup (server checks the ckpt; client checks the server via `/memory_config`):

```bash
TEMPORAL_MEMORY=false bash run_server.sh <model_epoch> /path/to/..._no_temporal_mem_run
TEMPORAL_MEMORY=false bash run_client.sh /path/to/..._no_temporal_mem_run <model_epoch>
```

Aggregate results — pass the **training-run root** (just like
`finetune/GemBench/summarize_eval.py`); the script discovers every
`eval/memorybench*/model_*/seed*/result.jsonl` under it and writes:

- `eval/<memorybench*>/summary.csv` — one row per (epoch, seed) with per
  base task and overall success rate;
- `eval/<memorybench*>/summary.txt` — same overview plus an epoch / base
  task / variant hierarchy;
- `eval/<memorybench*>/model_*/seed*/{result_detail,summary}.txt` — per-seed
  per-taskvar breakdown.

```bash
python summarize_eval.py /.../logs/train_memorybench/<run_name>
# optionally:  --trials 25  (episodes-per-taskvar threshold for the
#                            "incomplete" warning; defaults to client.py's 25)
```

Re-running `run_client.sh` against an existing `result.jsonl` is safe: the
summarizer detects duplicate rollouts (the chunk boundary is `episode_id`
resetting to 0) and pools their episodes per taskvar.

## Notes / caveats

- **Image resolution**: raw saved data is 128×128 and we keep it native —
  `IMAGE_SIZE=128`, so `MemoryBench_Dataset._resize_*` is a no-op and the
  emitted PCD is exactly what the simulator produced (no fabricated
  intermediate-depth points). The eval `RLBenchEnv` runs at 128 too. The MVT
  renderer projects to `mvt.img_size` regardless of input resolution. See the
  long rationale in `utils/peract_utils_memorybench.py`.
- **No grounding co-training** for MemoryBench: this is action-only
  finetuning; the loop trains `up_action` and nothing else.
- **`reset_to_demo` requires per-variation layout**: the released data ships
  in `all_variations/episodes/`. RLBench's `Environment.get_demos` looks for
  `variation<vid>/episodes/`. `scripts/reorg_data_for_eval.py` builds the
  expected layout via symlinks; without it the client falls back to
  `task.reset()`.
- **Episode count per task variation**: on the test split, each task has 25
  episodes by default (`--num_episodes`). Override with `NUM_EPISODES=` env
  var or `--num_episodes` directly to client.py.
- **Memory-only tasks need `memory.enabled=true`**: hard-coded in the
  shipped `memorybench_config.yaml`. Disabling it removes the anchor +
  history pathway, in which case the model cannot succeed on
  `put_block_back` (the policy has no way to remember the original color
  patch).
