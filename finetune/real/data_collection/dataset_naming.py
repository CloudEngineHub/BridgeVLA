"""Shared naming helpers for data collection and organize_dataset."""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

# Root of the raw real-robot collection tree. There is deliberately no default:
# it lives on a different disk on every machine, and silently falling back to a
# wrong tree either reads nothing or scatters episodes somewhere unexpected.
# Point it at YOUR OWN absolute path before running the collection / organize
# scripts:
#     export REAL_COLLECT_DATA_ROOT=/abs/path/to/your/real_collect
# Callers pass it through require_data_root(), which aborts with an explicit
# message while it is still unset.
DEFAULT_DATA_ROOT = os.environ.get("REAL_COLLECT_DATA_ROOT", "")


def require_data_root(
    path: str,
    env_var: str = "REAL_COLLECT_DATA_ROOT",
    cli_flag: str = "--data-root",
    must_exist: bool = True,
) -> str:
    """Return ``path``, or abort with instructions if it is unset / missing."""
    if not path:
        raise SystemExit(
            f"[config] no data root configured.\n"
            f"         Set it to your own absolute path, either\n"
            f"             export {env_var}=/abs/path/to/your/real_data\n"
            f"         or pass\n"
            f"             {cli_flag} /abs/path/to/your/real_data"
        )
    if must_exist and not os.path.isdir(path):
        raise SystemExit(
            f"[config] data root does not exist: {path}\n"
            f"         create it, or point {env_var} / {cli_flag} at the right tree"
        )
    return path


def require_file(path: str, env_var: str, cli_flag: str, what: str) -> str:
    """Return ``path``, or abort with instructions if it is unset / missing."""
    if not path:
        raise SystemExit(
            f"[config] no {what} configured.\n"
            f"         Set it to your own absolute path, either\n"
            f"             export {env_var}=/abs/path/to/your/file.npy\n"
            f"         or pass\n"
            f"             {cli_flag} /abs/path/to/your/file.npy"
        )
    if not os.path.isfile(path):
        raise SystemExit(
            f"[config] {what} not found: {path}\n"
            f"         point {env_var} / {cli_flag} at your own file"
        )
    return path


def instruction_to_slug(instruction: str) -> str:
    """Natural language instruction -> underscore slug (lowercase)."""
    text = instruction.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    parts = [p for p in re.split(r"[\s_]+", text) if p]
    return "_".join(parts)


def parse_episode_folder_name(folder_name: str) -> Optional[Tuple[str, str]]:
    """
    Parse flat episode folder: {task_slug}_{episode_idx}

    Examples:
        put_the_lids_on_the_blocks_then_uncover_the_blue_block_0
            -> ("put_the_lids_on_the_blocks_then_uncover_the_blue_block", "0")
        put_the_lids_on_the_blocks_then_uncover_the_red_block_12
            -> ("put_the_lids_on_the_blocks_then_uncover_the_red_block", "12")
    """
    name = folder_name.strip()
    m = re.match(r"^(.+)_(\d+)$", name)
    if not m:
        return None
    task_slug, episode_idx = m.group(1), m.group(2)
    if not task_slug:
        return None
    return task_slug, episode_idx


def list_episode_indices(data_root: str, task_slug: str) -> list[int]:
    """Return sorted episode indices already present for this task slug."""
    prefix = f"{task_slug}_"
    indices: list[int] = []
    if not os.path.isdir(data_root):
        return indices
    for name in os.listdir(data_root):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit() and os.path.isdir(os.path.join(data_root, name)):
            indices.append(int(suffix))
    return sorted(indices)


def find_next_episode_index(data_root: str, task_slug: str) -> int:
    """Next free episode index (0 if none exist)."""
    indices = list_episode_indices(data_root, task_slug)
    return max(indices) + 1 if indices else 0


def resolve_episode_save_path(data_root: str, instruction: str) -> str:
    """Absolute path for the next episode folder for this instruction."""
    task_slug = instruction_to_slug(instruction)
    idx = find_next_episode_index(data_root, task_slug)
    return os.path.join(data_root, f"{task_slug}_{idx}")


# Converted (dobot) episode index width: *_000, *_001, *_010, ...
OUTPUT_EPISODE_IDX_WIDTH = 3


def format_output_episode_idx(episode_idx: str | int) -> str:
    """Zero-pad episode index for converted output dirs (e.g. 10 -> '010')."""
    return f"{int(episode_idx):0{OUTPUT_EPISODE_IDX_WIDTH}d}"


def make_output_name(task_slug: str, episode_idx: str) -> str:
    """Converted output directory name: {task_slug}_{000|001|...}."""
    return f"{task_slug}_{format_output_episode_idx(episode_idx)}"
