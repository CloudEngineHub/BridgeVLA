# env_locks — reference freezes of the tested environments

`pip freeze` snapshots of the exact conda environments used to train and
evaluate the released BridgeVLA++ checkpoints:

| file | env | used by |
|---|---|---|
| `bridgevla_plus_rlbench.freeze.txt` | python 3.9, torch 2.5.1+cu121 | RLBench, Colosseum |
| `bridgevla_plus_gembench.freeze.txt` | python 3.9, torch 2.5.1+cu121 | GemBench, memoryBench, RMBench policy server, real-robot training, pretraining |
| `bridgevla_plus_rmbench.freeze.txt` | python 3.10, torch 2.4.1+cu121 | RMBench SAPIEN client |

A gembench env created by `install_gembench.sh --policy-only` (what
`install_rmbench.sh` sets up when you never installed full GemBench) is a
strict subset of the gembench freeze: the handful of packages pulled in by
the PyRep/RLBench requirement files are expected to be absent when diffing.

The rmbench freeze also records a few dev-time extras of the reference
machine (e.g. `scikit-image`, and the `scipy` 1.15.x it dragged in above the
`requirements.txt` pin of 1.10.1). A fresh install correctly produces the
pinned versions; the RMBench code only touches scipy APIs present in both.

These are **reference artifacts, not inputs**: the install scripts
(`finetune/*/install_*.sh`) resolve dependencies themselves and only pin what
matters (torch / xformers / transformers / setuptools / pip). If a future
transitive release breaks something, diff your env against the freeze here to
find the offending package:

```bash
pip freeze | diff - env_locks/bridgevla_plus_gembench.freeze.txt | less
```

Editable (`-e`) and local-path entries are omitted — in-repo code is imported
via each launch script's `PYTHONPATH`, and the RLBench/PyRep source stacks are
rebuilt at pinned commits by `scripts/fetch_sim_stacks.sh`.
