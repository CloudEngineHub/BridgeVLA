"""Memory-ablation switch helpers shared by the eval entry points
(GemBench server / RLBench eval). Mirrors the RMBench eval_config guard
(RMBench/policy/BridgeVLA_Plus/bridgevla_rmbench_model.py).

The two master switches live in the training yaml (memory.temporal_memory /
memory.spatial_memory) and are baked into the checkpoint's dumped
mvt_cfg.yaml. At eval the launcher declares which switches it EXPECTS
(TEMPORAL_MEMORY / SPATIAL_MEMORY in the .sh); if they disagree with the
loaded model we fail fast instead of silently evaluating the wrong ablation.

Effective model switch = memory.enabled AND <group switch>; checkpoints
predating the switches (keys absent in mvt_cfg.yaml) default to True, i.e.
full memory — identical to the RMBench convention.
"""


def parse_bool_flag(value, flag_name="flag"):
    """Parse a shell-style bool ('true'/'false'/'1'/'0'/'yes'/'no'/'on'/'off',
    case-insensitive; real bools pass through). Raises ValueError on anything
    else so a typo in the .sh aborts the run instead of silently meaning False.
    """
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on", "t"):
        return True
    if v in ("0", "false", "no", "off", "f"):
        return False
    raise ValueError(
        f"invalid bool for {flag_name}: {value!r} (expected true/false)"
    )


def model_memory_switches(mvt_cfg):
    """Effective (temporal, spatial) memory switches of a loaded model.

    ``mvt_cfg`` is the yacs node merged from the checkpoint's mvt_cfg.yaml.
    Effective switch = memory.enabled AND the group switch (missing keys
    default True: old checkpoints without the ablation keys are full-memory).
    """
    mem = getattr(mvt_cfg, "memory", None)
    enabled = bool(getattr(mem, "enabled", False)) if mem is not None else False
    temporal = enabled and bool(getattr(mem, "temporal_memory", True))
    spatial = enabled and bool(getattr(mem, "spatial_memory", True))
    return temporal, spatial


def assert_eval_memory_matches_model(mvt_cfg, eval_temporal, eval_spatial,
                                     where="eval"):
    """Fail fast if the eval-side memory switches disagree with the loaded
    model's memory config. The architecture is fixed at train time; this
    guards against evaluating the wrong ckpt for a memory ablation (e.g. eval
    declares temporal memory but the ckpt was trained without it).

      * temporal_memory -> memory 1, the stage-1 (mvt1) temporal memory + stage-1 anchor
      * spatial_memory  -> memory 2, the stage-2 (mvt2) spatial anchor
    """
    model_temporal, model_spatial = model_memory_switches(mvt_cfg)
    mem = getattr(mvt_cfg, "memory", None)
    enabled = bool(getattr(mem, "enabled", False)) if mem is not None else False

    mismatches = []
    if bool(eval_temporal) != model_temporal:
        mismatches.append(
            f"  - temporal_memory (memory 1, stage-1 temporal memory): "
            f"eval={bool(eval_temporal)} vs model={model_temporal} "
            f"(memory.enabled={enabled}, "
            f"temporal_memory={bool(getattr(mem, 'temporal_memory', True))})"
        )
    if bool(eval_spatial) != model_spatial:
        mismatches.append(
            f"  - spatial_memory (memory 2, stage-2 spatial memory): "
            f"eval={bool(eval_spatial)} vs model={model_spatial} "
            f"(memory.enabled={enabled}, "
            f"spatial_memory={bool(getattr(mem, 'spatial_memory', True))})"
        )
    if mismatches:
        raise ValueError(
            f"[{where}] eval memory switches do NOT match the loaded model's "
            "memory config:\n" + "\n".join(mismatches) + "\n"
            "Fix TEMPORAL_MEMORY / SPATIAL_MEMORY in the eval launcher to "
            "match the ckpt, or evaluate the matching ckpt."
        )
    print(f"[{where}] memory switches OK "
          f"(temporal={model_temporal}, spatial={model_spatial}).")
    return model_temporal, model_spatial
