# RLBench 587a6a0 stack patch bundle (the RLBench benchmark's training environment)

**What this is**: every RLBench-benchmark checkpoint was trained and evaluated on
`buttomnutstoast/RLBench@587a6a0` plus the 3 locally modified files in this directory (collected from the
training machine's actual working tree). This repository does not ship RLBench sources (its licence
forbids redistribution), so `install_rlbench.sh` clones the public upstream into
`finetune/bridgevla/libs/RLBench_peract587/` and rebuilds a byte-identical stack with this directory's
`apply_patch.sh`; `RLBench/train.sh` / `eval.sh` then pick it up via `RLBENCH_SIM_STACK`.

**Nature of the patch**: pure I/O optimisation plus a packaging fix (see "The three changes" below). It
**touches no simulation physics, action execution or success criterion** — scoring stays identical to the
upstream fork, which is what keeps the evaluation protocol the same as PerAct/RVT/original BridgeVLA. The
companion PyRep is `stepjam/PyRep@231a1ac` (unpatched). For an overview of both simulation stacks, see
`finetune/bridgevla/libs/rlbench_patches/README.md`.

---

This directory keeps the three locally modified files that the (gitignored)
`finetune/bridgevla/libs/RLBench_peract587/` needs, so the training environment can be reproduced anywhere.

## Fixed baseline

- Upstream repository: `https://github.com/buttomnutstoast/RLBench.git`
- Baseline commit: `587a6a0e6dc8cd36612a208724eb275fe8cb4470`
- Generated: 2026-07-29
- The file contents come from the uncommitted working tree at that baseline

The patch must be applied to that commit. Do not apply it to another RLBench version — the interfaces and
surrounding context may differ.

## Bundle contents

- `files/rlbench/environment.py`: the modified file in full
- `files/rlbench/utils.py`: the modified file in full
- `files/setup.py`: the modified file in full
- `rlbench-587a6a0-local.patch`: a unified diff of the three files against the fixed baseline
- `SHA256SUMS`: checksums of the files after applying
- `apply_patch.sh`: a migration script that verifies the baseline, the working tree and the result

## Recommended migration

Run from the repository root (usually handled by `install_rlbench.sh`, so you rarely need this by hand):

```bash
bash finetune/RLBench/rlbench_patch_bundle/apply_patch.sh \
  finetune/bridgevla/libs/RLBench_peract587
```

With the target argument omitted the script defaults to `finetune/bridgevla/libs/RLBench`; for this
repository's standard layout, pass `RLBench_peract587` explicitly. The script will:

1. confirm the target is a Git repository;
2. confirm HEAD is the fixed baseline commit;
3. confirm the three target files carry no other local modifications;
4. run `git apply --check` before applying the patch;
5. verify the resulting files with SHA-256.

If the three files already match this bundle, the script exits successfully without re-applying.

## Manual overwrite

Only use a full-file overwrite when you are sure the target should adopt this machine's versions wholesale:

```bash
BUNDLE=finetune/RLBench/rlbench_patch_bundle
TARGET=finetune/bridgevla/libs/RLBench_peract587

install -m 0644 "$BUNDLE/files/rlbench/environment.py" \
  "$TARGET/rlbench/environment.py"
install -m 0644 "$BUNDLE/files/rlbench/utils.py" \
  "$TARGET/rlbench/utils.py"
install -m 0644 "$BUNDLE/files/setup.py" \
  "$TARGET/setup.py"

(cd "$TARGET" && sha256sum -c "$OLDPWD/$BUNDLE/SHA256SUMS")
```

A manual overwrite does not protect modifications already present on the target machine, so prefer the
migration script.

## The three changes

### 1. `rlbench/environment.py`

Adds an optional parameter to `Environment.get_demos()`:

```python
load_images: bool = True
```

and forwards it to `utils.get_stored_demos()`, so callers can ask for low-dimensional observations only.

### 2. `rlbench/utils.py`

Adds the same parameter to `get_stored_demos()`. With `load_images=False` it:

- keeps the observations loaded from `low_dim_obs.pkl`;
- writes the language descriptions into each observation's `misc['descriptions']`;
- skips reading and converting the RGB, depth, mask and point-cloud files.

That path is what evaluation code such as GemBench's calls with `load_images=False`.

### 3. `setup.py`

Adds `rlbench.action_modes` to setuptools' `packages` list, so a non-editable install does not miss the
subpackage.

## Files deliberately not in this patch

The `.py` and `.ttm` files of the three MemoryBench tasks — `put_block_back`, `rearrange_block` and
`reopen_drawer` — are not here. They are pinned in-repo under
`finetune/bridgevla/libs/rlbench_patches/tasks/` (the versions every released checkpoint was trained and
evaluated with) and installed by `finetune/memoryBench/scripts/install_memorybench_tasks.sh`, which
overwrites any differing copy with the pinned one. Note that `data/files/` in the upstream data repository
hqfang/memorybench has since evolved and is no longer byte-identical to the pinned version — evaluation
always uses the in-repo pin.
