# Simulation source stacks and patches (canonical document)

This repository **does not ship** PyRep / RLBench sources (RLBench's Imperial College licence forbids
redistribution — the reason the PerAct/RVT community always "clones and builds" them). Rebuilding goes
through a single entry point, `scripts/fetch_sim_stacks.sh`, which clones the public upstreams at pinned
commits and applies the in-repo patches (idempotent, safe to re-run). Both installers —
`finetune/RLBench/install_rlbench.sh` (two stacks) and `finetune/GemBench/install_gembench.sh` (shared
stack) — call it, reproducing the source stacks byte-for-byte as used for training. Everything is
consumed through each `.sh`'s `PYTHONPATH`; there is no `pip install -e`.

## ⚠️ Two stacks coexist, they are version sensitive, and they must not be mixed

The two forks differ in success criteria, action execution and task scene files, so **evaluation must use
the same stack as training**. `RLBench/train.sh` / `eval.sh` put the right stack first via
`RLBENCH_SIM_STACK` / `PYREP_SIM_STACK`; a wrong or missing stack fails loudly rather than silently
falling back.

| Directory (all gitignored) | Upstream @ pin | Patches | Benchmarks served |
|---|---|---|---|
| `RLBench` | [rjgpinel/RLBench](https://github.com/rjgpinel/RLBench) @ `ebdc339` | this directory (4 files + 6 tasks) | GemBench / memoryBench / Colosseum |
| `PyRep` | [cshizhe/PyRep](https://github.com/cshizhe/PyRep) @ `7962b0e` | none | the same (its companion) |
| `RLBench_peract587` | [buttomnutstoast/RLBench](https://github.com/buttomnutstoast/RLBench) @ `587a6a0` | `finetune/RLBench/rlbench_patch_bundle/` (3 files) | **the RLBench benchmark** (the train/eval environment of its ckpt) |
| `PyRep_stepjam231` | [stepjam/PyRep](https://github.com/stepjam/PyRep) @ `231a1ac` | none | **the RLBench benchmark** (its companion) |

The split follows the environments that produced each set of paper numbers: the RLBench benchmark was
trained and evaluated on (stepjam PyRep + 587a6a0 RLBench), and the other CoppeliaSim benchmarks on
(cshizhe PyRep + rjgpinel RLBench).

## This directory: the rjgpinel stack patch (`rlbench_rjgpinel_ebdc339.patch`, 4 files, ~50 lines)

Purpose: make the GemBench-family fork compatible with PerAct-style data layouts and model interfaces.
**It touches no success criterion and no physics behaviour.**

| File | Change | Why it is needed |
|---|---|---|
| `rlbench/utils.py` | `get_stored_demos(variation_number=-1)` also accepts PerAct's flat `all_variations/episodes` layout and recovers the real variation from each episode's `variation_number.pkl` | The dataset uses the PerAct layout; without this every eval episode resets to variation 0 (wrong scene and wrong instruction) |
| `rlbench/backend/const.py` | Adds the constants `VARIATIONS_ALL_FOLDER` / `VARIATION_NUMBER` | A dependency of the row above |
| `rlbench/action_modes/arm_action_modes.py` | `EndEffectorPoseViaPlanning.action()` gains a per-step `ignore_collisions` argument (falling back to the constructor's behaviour when omitted) | RVT/PerAct-style models predict a collision flag every step and must pass it to the motion planner; this fork only had a global switch |
| `rlbench/backend/observation.py` | Live observations gain an `ignore_collisions` field (default 1.0) | YARR/peract_colab require the field to exist when reading observations; at eval it is an inert placeholder the policy never reads |

`tasks/` holds the 6 task definitions (`.py` + `.ttm`) this fork is missing: 3 MemoryBench tasks
(`put_block_back` / `rearrange_block` / `reopen_drawer`, byte-identical to the dataset release) and
3 PerAct tasks (`place_wine_at_rack_location` / `slide_block_to_color_target` /
`sweep_to_dustpan_of_size`) that RLBench-18 needs.

## The 587a6a0 stack patch

See `finetune/RLBench/rlbench_patch_bundle/README.md` (an `get_demos(load_images=)` I/O optimisation plus
a `setup.py` packaging fix; it likewise touches no scoring logic).

## Manual rebuild (for reference/audit only; normally just run the first line)

```bash
bash scripts/fetch_sim_stacks.sh --all    # clone + checkout + patch, idempotent
# Equivalent manual steps:
cd finetune/bridgevla/libs

# Shared stack (GemBench / memoryBench / Colosseum)
git clone https://github.com/cshizhe/PyRep.git   && git -C PyRep   checkout 7962b0e04700315c2b0de87a994dbfe77c915c17
git clone https://github.com/rjgpinel/RLBench.git && git -C RLBench checkout ebdc3392c1a11c4cdcc9a440cd61ec345bef42ec
git -C RLBench apply ../libs/rlbench_patches/rlbench_rjgpinel_ebdc339.patch  # (relative to this directory)
cp rlbench_patches/tasks/*.py  RLBench/rlbench/tasks/
cp rlbench_patches/tasks/*.ttm RLBench/rlbench/task_ttms/

# RLBench benchmark stack
git clone https://github.com/stepjam/PyRep.git PyRep_stepjam231 && git -C PyRep_stepjam231 checkout 231a1ac6b0a179cff53c1d403d379260b9f05f2f
git clone https://github.com/buttomnutstoast/RLBench.git RLBench_peract587 && git -C RLBench_peract587 checkout 587a6a0e6dc8cd36612a208724eb275fe8cb4470
bash ../../RLBench/rlbench_patch_bundle/apply_patch.sh RLBench_peract587   # verifies baseline + SHA256 three ways

# Build the C extensions in place (export COPPELIASIM_ROOT and LD_LIBRARY_PATH first;
# install_rlbench.sh / install_gembench.sh do this inside the conda env)
(cd PyRep            && python setup.py build_ext --inplace)
(cd PyRep_stepjam231 && python setup.py build_ext --inplace)
```
