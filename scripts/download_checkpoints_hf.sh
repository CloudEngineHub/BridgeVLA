#!/usr/bin/env bash
# download_checkpoints_hf.sh — released BridgeVLA++ artifacts, from HuggingFace
# (https://huggingface.co/datasets/LPY/BridgeVLA).
#
# Same targets and options as the ModelScope variant (download_checkpoints_ms.sh);
# `-h` lists them. Note: the `paligemma` target is gated on HuggingFace (accept
# the license + `hf auth login`) — the _ms variant pulls an ungated mirror.
BRIDGEVLA_DL_SOURCE=hf exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_download_checkpoints_impl.sh" "$@"
