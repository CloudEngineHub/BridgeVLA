#!/usr/bin/env bash
# download_checkpoints_ms.sh — released BridgeVLA++ artifacts, from ModelScope
# (https://modelscope.cn/models/susetiankong/bridgevla_plus — identical mirror
# of the HuggingFace release).
#
# Same targets and options as the HuggingFace variant (download_checkpoints_hf.sh);
# `-h` lists them. The `paligemma` target uses the ungated mirror
# AI-ModelScope/paligemma-3b-pt-224 (no login needed).
# Requires the `modelscope` CLI: pip install modelscope
BRIDGEVLA_DL_SOURCE=ms exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_download_checkpoints_impl.sh" "$@"
