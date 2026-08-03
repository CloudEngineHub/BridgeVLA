# Scripts & reference notes

The helper scripts in this directory each carry a detailed usage header — read that first:

| script | purpose |
|---|---|
| `download_checkpoints_hf.sh` / `download_checkpoints_ms.sh` | released weights & pre-training corpus — identical content from HuggingFace / ModelScope, pick by network; thin wrappers over `_download_checkpoints_impl.sh` (the shared engine, not run directly). `--list` shows every target + size, `--config-only` peeks at architecture configs before a large pull |
| `download_datasets.sh` | third-party benchmark datasets (HuggingFace-only upstreams), into the exact layout the train/eval scripts expect (`--extract` unpacks / prepares); its release-hosted parts (keyframe caches, RMBench keyframe data) delegate to the checkpoint downloader — `BRIDGEVLA_DL_SOURCE=ms` pulls those from ModelScope |
| `fetch_sim_stacks.sh` | rebuild the pinned RLBench/PyRep source stacks + patches (called by the installers) |
| `build_rlbench_cache.sh` | rebuild the RLBench keyframe cache locally instead of downloading the released one |

The rest of this file collects reference material deliberately kept out of the top-level README.

## Repository layout

```
├── env_locks/                # pip freezes of the exact tested environments (reference)
├── finetune/
│   ├── bridgevla/            # core model (mvt / models / utils + vendored libs)
│   │   └── libs/             #   peract, peract_colab, point-renderer, YARR (vendored)
│   │       └── rlbench_patches/  # patches + pinned task files for the sim stacks
│   ├── Colosseum/  GemBench/  memoryBench/  RLBench/   # 4 CoppeliaSim benchmarks
│   ├── real/                 # real robot: collection / training / deployment (Dobot + ZED)
│   ├── RMBench/              # SAPIEN bimanual benchmark (RoboTwin-Platform/RMBench fork)
│   └── RMBench_vla/          # BridgeVLA++ training side for RMBench
├── pretrain/                 # grounding pre-training on the RoboPoint corpus (PaliGemma)
└── scripts/                  # this directory
```

## Conda environments

Each installer creates its own environment (details and pins in each script's header):

| conda env (default name) | created by | Python / PyTorch | notes |
|---|---|---|---|
| `bridgevla_plus_rlbench` | `finetune/RLBench/install_rlbench.sh` | 3.9 / 2.5.1+cu121 | RLBench + COLOSSEUM; also clones+patches the sim sources and builds a C/CUDA extension |
| `bridgevla_plus_gembench` | `finetune/GemBench/install_gembench.sh` | 3.9 / 2.5.1+cu121 | GemBench + memoryBench + the RMBench policy side; a *different* CoppeliaSim stack than RLBench (see below). `--policy-only` installs just the pip stack + point-renderer (no apt/simulator steps) — what `install_rmbench.sh` runs for you |
| `bridgevla_plus_rmbench` | `finetune/RMBench/install_rmbench.sh` | 3.10 / 2.4.1+cu121 | SAPIEN simulation client (its STEP 0 also creates the policy env above, in `--policy-only` mode); patches two site-packages files (sapien, mplib) |
| `bridgevla_plus_pretrain` | `pretrain/install_pretrain.sh` | 3.9 / 2.5.1+cu121 | strict subset of the GemBench env — no simulator |
| `bridgevla_plus_real_train` | `finetune/real/install_real_train.sh` | 3.9 / 2.5.1+cu121 | strict subset of the GemBench env — GPU training server |
| `bridgevla_plus_real_deploy` | `finetune/real/install_real_deploy.sh` | 3.10 / 2.6.0+cu124 | robot workstation only |

**RMBench needs two envs** because it runs as a client/server pair: the SAPIEN simulator (`script/eval_policy_client.py`) lives in the rmbench env, the BridgeVLA++ policy server in the shared policy env. `install_rmbench.sh` sets up both — it first runs `install_gembench.sh --policy-only` (pip stack + point-renderer build; no CoppeliaSim / RLBench / PyRep), then builds the SAPIEN env — so RMBench never requires a GemBench simulator install. The eval launcher activates the right env per side itself.

Notes:

- The four py3.9 environments share the same core pins (torch 2.5.1+cu121, transformers 4.51.3, xformers 0.0.28.post3); the two benchmark envs differ only in which simulation *source stack* the launch scripts put on `PYTHONPATH`.
- **Fewer envs if you prefer** — every installer and train/eval script honours `RLBENCH_CONDA_ENV` / `GEMBENCH_CONDA_ENV` / `RMBENCH_CONDA_ENV` / `PRETRAIN_CONDA_ENV` / `REAL_TRAIN_CONDA_ENV`. E.g. run pre-training inside the GemBench env and skip its installer: `PRETRAIN_CONDA_ENV=bridgevla_plus_gembench bash pretrain/pretrain.sh`; or install both CoppeliaSim families into one env by running both installers with `RLBENCH_CONDA_ENV` and `GEMBENCH_CONDA_ENV` set to the same name.
- `env_locks/*.freeze.txt` are `pip freeze` snapshots of the exact environments the released checkpoints were produced with — diff against them when debugging a version issue.
- No `pip install -e` anywhere (single exception: RMBench's curobo, whose CUDA extension requires it). All in-repo code resolves through each script's `PYTHONPATH`; the envs contain no absolute path into the checkout, so moving or duplicating the repo is safe.

## `data/` layout

`data/` is gitignored and may be a symlink; the two roots can be moved elsewhere with `BRIDGEVLA_DATA_ROOT` / `BRIDGEVLA_CKPT_ROOT`. After the downloads:

```
data/
├── bridgevla_ckpt/
│   ├── paligemma-3b-pt-224/          # base VLM
│   ├── clip/RN50.pt                  # CLIP visual encoder (peract, offline load)
│   ├── pretrain/                     # grounding warm start (HF sharded safetensors)
│   ├── bridgevla_plus/               # released BridgeVLA++ checkpoints
│   │   ├── rlbench/ colosseum/ gembench/ memorybench/
│   │   └── rmbench/<task>/           #   per-task (9 tasks)
│   └── bridgevla/                    # legacy BridgeVLA checkpoints (original codebase only)
└── bridgevla_data/
    ├── RLBench/                      # demos + _keyframe_cache/size128_v2 (released, REQUIRED)
    ├── Colosseum/  GemBench/
    ├── memorybench/data/{train,test} #   train/../_keyframe_cache/size128_v3 (released)
    ├── RMBench/
    │   ├── assets/                   # upstream sim assets (eval)
    │   └── data/{keyframe_data,keyframes}/   # released training HDF5 + keyframe metadata
    │                                 #   (keyframes/ MUST stay next to keyframe_data/)
    ├── pretrain_data/                # RoboPoint corpus (only for re-running pretraining)
    ├── Real/<collection>/            # self-collected real-robot data
    └── logs/                         # training outputs (train / train_gembench / ...)
```

## Benchmark dataset upstreams

| target | upstream data | size | destination |
|---|---|---|---|
| `rlbench` | [hqfang/rlbench-18-tasks](https://huggingface.co/datasets/hqfang/rlbench-18-tasks) (PerAct demos, 100 train + 25 test ep/task) | 116 GiB | `data/bridgevla_data/RLBench` |
| `gembench` | [rjgpinel/GEMBench](https://huggingface.co/datasets/rjgpinel/GEMBench) (`keysteps_bbox` train + `microsteps` test; `GEMBENCH_FULL=1` for the whole repo) | 162 GiB | `data/bridgevla_data/GemBench` |
| `memorybench` | [hqfang/memorybench](https://huggingface.co/datasets/hqfang/memorybench) (SAM2Act's 3 memory tasks) | 22 GiB | `data/bridgevla_data/memorybench` |
| `colosseum` | [colosseum/colosseum-challenge](https://huggingface.co/datasets/colosseum/colosseum-challenge) training archives | 75 GiB | `data/bridgevla_data/Colosseum` |
| `colosseum_eval` | same repo, per-task variation archives (subset via `COLOSSEUM_EVAL_TASKS`) | 186 GiB | `data/bridgevla_data/Colosseum` |
| `rmbench` | [TianxingChen/RMBench](https://huggingface.co/datasets/TianxingChen/RMBench) assets (eval sim) + BridgeVLA++ release keyframe data (training); raw `demo_clean` demos only with `RMBENCH_RAW=1` | ~31 GiB (+37 GiB raw) | `data/bridgevla_data/RMBench` |

memoryBench note: the three task definitions (`.py`/`.ttm`) are **pinned inside this repo** (`finetune/bridgevla/libs/rlbench_patches/tasks/`) and installed automatically; the copies in the upstream dataset have drifted since our training and are intentionally not used.

## Keyframe indexes & caches

Keyframe supervision is precomputed per benchmark. What ships in the release vs. what builds automatically on first run:

| bench | artifact | first run |
|---|---|---|
| RLBench | `RLBench/_keyframe_cache/size128_v2/<task>/episode<N>.npz+.meta` — canonical **majority-vote** keyframes (per task+variation) + decoded obs | **shipped, required** (`download_checkpoints_hf.sh rlbench_cache`). Training loads it strictly and never rebuilds; the `.meta` keyframes stay authoritative even if you delete the `.npz` (re-rendered from the meta, not re-voted). Local rebuild from raw demos (hours; the vote runs over your local episode set, so results can differ from the released cache): `bash scripts/build_rlbench_cache.sh` |
| COLOSSEUM | same layout under `Colosseum/_keyframe_cache/size128_v2` | built by `prepare_colosseum_data.sh` during extraction (10 parallel workers) |
| memoryBench | `memorybench/data/train/_keyframe_cache/size128_v3` | shipped (`download_checkpoints_hf.sh memorybench_cache`); would otherwise build lazily on first run |
| GemBench | tiny JSON transition index under `train_dataset/cache/` | auto-built in seconds (upstream data is already keyframe-only) |
| RMBench | `RMBench/data/keyframes/<task>.json` (keyframe indices + per-keyframe subtask annotation → memory labels) + `RMBench/data/keyframe_data/` (keyframe-only HDF5) | **shipped, required** (`download_checkpoints_hf.sh rmbench_data`); regenerable from the raw demos — see below |

All loaders **fail fast** on a missing/stale/mismatched cache (including a camera-order mismatch) instead of silently degrading or shrinking the dataset. The RLBench/Colosseum `train.sh` additionally check the cache **before** launching torchrun, so a missing cache yields one clean message with the fix commands instead of a traceback per DDP rank.

Regenerating the RMBench keyframe data from the raw `demo_clean` demos (`RMBENCH_RAW=1 bash scripts/download_datasets.sh rmbench`), in the rmbench env:

```bash
# 1) keyframe metadata jsons -> RMBench/data/keyframes/<task>.json
python finetune/RMBench/script/extract_key_frames/extract_keyframes.py \
    --data_root data/bridgevla_data/RMBench/data/data \
    --output_dir data/bridgevla_data/RMBench/data/keyframes
# 2) keyframe-only HDF5 (SAPIEN re-render) -> RMBench/data/keyframe_data/
bash finetune/RMBench/collect/batch/run_all_keyframe_depth.sh   # or per task:
bash finetune/RMBench/collect/single/run_keyframe_depth.sh press_button 0
```

## Simulation source stacks

`fetch_sim_stacks.sh` clones pinned upstream commits and applies this repo's patches, rebuilding the exact source trees the released checkpoints were trained with:

| stack (under `finetune/bridgevla/libs/`, gitignored) | upstream @ pin | patch | used by |
|---|---|---|---|
| `RLBench` + `PyRep` | [rjgpinel/RLBench](https://github.com/rjgpinel/RLBench) @ `ebdc339`, [cshizhe/PyRep](https://github.com/cshizhe/PyRep) @ `7962b0e` | `bridgevla/libs/rlbench_patches/` | GemBench, memoryBench, COLOSSEUM |
| `RLBench_peract587` + `PyRep_stepjam231` | [buttomnutstoast/RLBench](https://github.com/buttomnutstoast/RLBench) @ `587a6a0`, [stepjam/PyRep](https://github.com/stepjam/PyRep) @ `231a1ac` | `finetune/RLBench/rlbench_patch_bundle/` (SHA-256 verified) | **RLBench benchmark** |

> ⚠️ **The stacks are not interchangeable** — a checkpoint must be evaluated on the stack it was trained with. `RLBench/train.sh` and `eval.sh` select the correct stack automatically via `RLBENCH_SIM_STACK` / `PYREP_SIM_STACK` and fail loudly if it is missing.

Patch contents and rationale: `finetune/bridgevla/libs/rlbench_patches/README.md` (canonical overview) and `finetune/RLBench/rlbench_patch_bundle/README.md`. Set `RLBENCH_UPSTREAM` / `PYREP_UPSTREAM` / `RLBENCH_PERACT_UPSTREAM` / `PYREP_STEPJAM_UPSTREAM` to your own forks as insurance against upstream deletion.

## Troubleshooting

- **RLBench scores look destroyed** → you are on the wrong simulation stack (see above). `RLBench/train.sh`/`eval.sh` wire the correct one automatically — do not bypass them without exporting `RLBENCH_SIM_STACK`/`PYREP_SIM_STACK` equivalently.
- **Suspected wrong package/import resolution** — imports are governed purely by each script's `PYTHONPATH` (no `pip install -e`): `python -c "import rlbench, pyrep; print(rlbench.__file__, pyrep.__file__)"` must resolve inside this repository.
- **GemBench/memoryBench eval hangs** → server and client must share the same `PORT` (defaults 13130 / 13168) and run on different GPUs.
- **CoppeliaSim/pyrep import fails with `libcoppeliaSim.so.1` not found** → the launch scripts export `COPPELIASIM_ROOT` + `LD_LIBRARY_PATH` themselves; for manual python sessions replicate the exports from any `train.sh`.
