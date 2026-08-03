"""Machine-independent path roots for the real-robot pipeline.

Single source of truth for "where does the data / logs live", so no .py or
.yaml under ``real/`` needs a literal absolute path.

Resolution order for each root:
  1. the environment variable exported by the launcher .sh
     (``real/train.sh`` exports them from the repo location);
  2. otherwise derive from this file's location. ``paths.py`` lives at
     ``<BRIDGEVLA_ROOT>/finetune/real/paths.py``, so the derivation is
     correct even when a script is run bare with ``python real/foo.py``
     and no .sh wrapper.

``expand()`` additionally lets YAML configs carry ``${BRIDGEVLA_DATA_ROOT}/...``
placeholders (same convention as GemBench's gembench_config.yaml) and resolves
them even when the env var is unset — plain ``os.path.expandvars`` would leave
the literal ``${...}`` in the path and fail much later with a confusing ENOENT.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../finetune/real
FINETUNE_DIR = os.path.dirname(_THIS_DIR)                # .../finetune
_DERIVED_ROOT = os.path.dirname(FINETUNE_DIR)            # .../BridgeVLA_plus

BRIDGEVLA_ROOT = os.environ.get("BRIDGEVLA_ROOT") or _DERIVED_ROOT
BRIDGEVLA_DATA_ROOT = (os.environ.get("BRIDGEVLA_DATA_ROOT")
                       or os.path.join(BRIDGEVLA_ROOT, "data", "bridgevla_data"))
BRIDGEVLA_CKPT_ROOT = (os.environ.get("BRIDGEVLA_CKPT_ROOT")
                       or os.path.join(BRIDGEVLA_ROOT, "data", "bridgevla_ckpt"))

# --- real-pipeline roots ---------------------------------------------------
REAL_DATA_ROOT = (os.environ.get("REAL_DATA_ROOT")
                  or os.path.join(BRIDGEVLA_DATA_ROOT, "Real"))
REAL_LOG_DIR = (os.environ.get("BRIDGEVLA_REAL_LOG_DIR")
                or os.path.join(BRIDGEVLA_DATA_ROOT, "logs_real"))
PRETRAIN_LOG_DIR = os.path.join(BRIDGEVLA_DATA_ROOT, "logs", "pretrain")

# Placeholders honoured by expand(). Longest name first so that replacing
# "$BRIDGEVLA_ROOT" can never eat the prefix of "$BRIDGEVLA_DATA_ROOT".
_PLACEHOLDERS = (
    ("BRIDGEVLA_DATA_ROOT", BRIDGEVLA_DATA_ROOT),
    ("BRIDGEVLA_CKPT_ROOT", BRIDGEVLA_CKPT_ROOT),
    ("BRIDGEVLA_REAL_LOG_DIR", REAL_LOG_DIR),
    ("BRIDGEVLA_ROOT", BRIDGEVLA_ROOT),
    ("REAL_DATA_ROOT", REAL_DATA_ROOT),
)


def expand(path):
    """Expand ``${VAR}`` / ``~`` in a config path, with our roots as fallback.

    Returns non-str input unchanged so callers can pass through None/lists.
    """
    if not isinstance(path, str) or not path:
        return path
    for var, val in _PLACEHOLDERS:
        path = path.replace("${%s}" % var, val).replace("$%s" % var, val)
    return os.path.expanduser(os.path.expandvars(path))
