#!/usr/bin/env python3
"""Summarize Colosseum success rates for every model and eval run.

Ported from RLBench/summarize_eval.py. The output contains one CSV per
``model_*`` directory. Tasks (rows) carry the ``<base_task>_<variation>``
suffix used by Colosseum eval; rows are ordered by base task then variation
index. Eval-run directories (columns) are the ``EVAL_LOG_NAME`` dirs written
by eval.sh (default ``YYYYmmdd_HHMMSS_var<N>``); dirs without a result CSV are
retained as empty columns.
"""

from __future__ import annotations

import csv
import re
import sys
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ======================== Edit this path ========================
# The ckpt directory to summarise: eval results land under its eval/<model_stem>/<log_name>/
# (see eval.sh). Released ckpt directories and your own training runs share the same layout, so both
# can be summarised directly; point COLOSSEUM_SUMMARY_DIR at one instead of editing this file.
TRAIN_RUN_DIR = Path(
    os.environ.get("COLOSSEUM_SUMMARY_DIR")
    or (os.environ.get("BRIDGEVLA_RELEASE_CKPT_DIR", "data/bridgevla_ckpt/bridgevla_plus") + "/colosseum")
)


COLOSSEUM_TASKS = [
    "basketball_in_hoop",
    "close_box",
    "empty_dishwasher",
    "get_ice_from_fridge",
    "hockey",
    "meat_on_grill",
    "move_hanger",
    "wipe_desk",
    "open_drawer",
    "slide_block_to_target",
    "reach_and_drag",
    "put_money_in_safe",
    "place_wine_at_rack_location",
    "insert_onto_square_peg",
    "turn_oven_on",
    "straighten_rope",
    "setup_chess",
    "scoop_with_spatula",
    "close_laptop_lid",
    "stack_cups",
]

MODEL_DIR_RE = re.compile(r"^model_(\d+|last)(?:_(.*))?$")
# eval.sh's EVAL_LOG_NAME defaults: YYYYmmdd_HHMMSS_var<N> (single variation) or
# YYYYmmdd_HHMMSS_sweep<N>vars (a multi-variation sweep); a bare timestamp and arbitrary custom
# suffixes are accepted too (the user may set EVAL_LOG_NAME to anything). Anything starting with an
# 8-digit date (optionally _HHMMSS) counts as an eval-run directory; model_*/colosseum_summary etc. do not start with a date and are excluded naturally.
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}(?:_\d{6})?(?:_.+)?$")
TASK_VARIATION_RE = re.compile(r"^(?P<base>.+)_(?P<var>\d+)$")


def model_sort_key(path: Path) -> tuple[int, str]:
    match = MODEL_DIR_RE.fullmatch(path.name)
    if match is None:
        return (sys.maxsize, path.name)
    idx = match.group(1)
    return (sys.maxsize - 1 if idx == "last" else int(idx), match.group(2) or "")


def find_eval_dir(run_dir: Path) -> Path:
    """Accept either a training run directory or its eval directory."""
    if run_dir.name == "eval":
        return run_dir
    return run_dir / "eval"


def parse_success_rates(csv_paths: Iterable[Path]) -> Dict[str, float]:
    """Read task success rates; later duplicate task rows take precedence."""
    success_rates: Dict[str, float] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            required = {"task", "success rate"}
            if not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Missing columns {sorted(required)} in {csv_path}; "
                    f"found {reader.fieldnames}"
                )

            for row in reader:
                task = (row.get("task") or "").strip()
                value = (row.get("success rate") or "").strip()
                if not task or not value:
                    continue
                try:
                    success_rates[task] = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid success rate {value!r} for task {task!r} "
                        f"in {csv_path}"
                    ) from exc
    return success_rates


def task_sort_key(task: str) -> tuple[int, int, str]:
    """Order rows by base task (COLOSSEUM_TASKS order) then variation index.

    ``basketball_in_hoop_3`` -> (idx(basketball_in_hoop), 3, ...); names that
    don't match any known base task sort after everything, alphabetically.
    """
    match = TASK_VARIATION_RE.fullmatch(task)
    candidates = [task] + ([match.group("base")] if match else [])
    for base in candidates:
        if base in COLOSSEUM_TASKS:
            var = int(match.group("var")) if (match and base != task) else -1
            return (COLOSSEUM_TASKS.index(base), var, task)
    return (len(COLOSSEUM_TASKS), -1, task)


def ordered_tasks(results_by_timestamp: Dict[str, Dict[str, float]]) -> List[str]:
    observed = {
        task
        for timestamp_results in results_by_timestamp.values()
        for task in timestamp_results
    }
    return sorted(observed, key=task_sort_key)


# ---- The paper-protocol (Tab. colosseum) COLOSSEUM average ----
# variation 0 = no_variations = the in-distribution clean baseline, reference only and excluded from the average.
# variations 1..14 = the paper's 14 generalization settings. The COLOSSEUM average takes each
# setting's column mean across tasks first, then averages those 14 column means with equal weight
# (mean of column means). That matches how the original 64.0% is computed; averaging all task_vars
# directly would fold var0 in and weight by cell, giving a higher number. See colosseum_matrix.py.
PAPER_SETTINGS = list(range(1, 15))  # 1..14
BASELINE_VAR = 0  # no_variations


def variation_col_means(success_rates: Dict[str, float]) -> Dict[int, float]:
    """{'<task>_<var>': rate} -> {var: column mean across tasks}."""
    by_var: Dict[int, List[float]] = {}
    for task, rate in success_rates.items():
        match = TASK_VARIATION_RE.fullmatch(task)
        if match is None:
            continue
        by_var.setdefault(int(match.group("var")), []).append(rate)
    return {var: sum(rates) / len(rates) for var, rates in by_var.items() if rates}


def paper_average(success_rates: Dict[str, float]) -> Optional[float]:
    """COLOSSEUM average = the column means of the 1..14 settings present, averaged with equal weight (var0 excluded)."""
    col_means = variation_col_means(success_rates)
    reported = [col_means[var] for var in PAPER_SETTINGS if var in col_means]
    return sum(reported) / len(reported) if reported else None


def original_baseline(success_rates: Dict[str, float]) -> Optional[float]:
    """Column mean of no_variations (var0) — reference only, excluded from the COLOSSEUM average."""
    return variation_col_means(success_rates).get(BASELINE_VAR)


def collect_model_results(
    model_dir: Path,
) -> tuple[List[str], Dict[str, Dict[str, float]], List[str]]:
    timestamp_dirs = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_dir() and TIMESTAMP_DIR_RE.fullmatch(path.name)
    )

    timestamps: List[str] = []
    results_by_timestamp: Dict[str, Dict[str, float]] = {}
    empty_results: List[str] = []

    for timestamp_dir in timestamp_dirs:
        timestamp = timestamp_dir.name
        csv_paths = sorted(timestamp_dir.glob("eval_results*.csv"))
        timestamps.append(timestamp)
        success_rates = parse_success_rates(csv_paths)
        results_by_timestamp[timestamp] = success_rates
        if not success_rates:
            empty_results.append(timestamp)

    return timestamps, results_by_timestamp, empty_results


def write_model_csv(
    output_path: Path,
    timestamps: List[str],
    results_by_timestamp: Dict[str, Dict[str, float]],
) -> int:
    tasks = ordered_tasks(results_by_timestamp)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", *timestamps])

        for task in tasks:
            writer.writerow(
                [
                    task,
                    *[
                        results_by_timestamp[timestamp].get(task, "")
                        for timestamp in timestamps
                    ],
                ]
            )

        # The paper-protocol COLOSSEUM average (14 settings, var0 excluded).
        writer.writerow(
            [
                "COLOSSEUM avg (14 settings)",
                *[
                    "" if value is None else round(value, 2)
                    for value in (
                        paper_average(results_by_timestamp[timestamp])
                        for timestamp in timestamps
                    )
                ],
            ]
        )
        # The no_variations clean baseline: reference only, excluded from the average above.
        writer.writerow(
            [
                "Original (v0, not in avg)",
                *[
                    "" if value is None else round(value, 2)
                    for value in (
                        original_baseline(results_by_timestamp[timestamp])
                        for timestamp in timestamps
                    )
                ],
            ]
        )
    return len(tasks)


def summarize(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    eval_dir = find_eval_dir(run_dir)
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Colosseum eval directory does not exist: {eval_dir}")

    model_dirs = sorted(
        (
            path
            for path in eval_dir.iterdir()
            if path.is_dir() and MODEL_DIR_RE.fullmatch(path.name)
        ),
        key=model_sort_key,
    )
    if not model_dirs:
        raise FileNotFoundError(f"No model_* directories found in: {eval_dir}")

    output_dir = eval_dir / "colosseum_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for model_dir in model_dirs:
        timestamps, results_by_timestamp, empty_results = collect_model_results(
            model_dir
        )
        if not timestamps:
            print(f"Skip {model_dir.name}: no eval-run directories")
            continue

        output_path = output_dir / f"{model_dir.name}_summary.csv"
        task_count = write_model_csv(output_path, timestamps, results_by_timestamp)
        generated += 1
        print(
            f"{model_dir.name}: {len(timestamps)} eval runs, "
            f"{task_count} tasks -> {output_path}"
        )
        if empty_results:
            print(
                "  Empty columns (no usable success rates): "
                + ", ".join(empty_results)
            )

    if generated == 0:
        raise RuntimeError(f"No model summaries were generated from: {eval_dir}")

    print(f"Generated {generated} model summaries in: {output_dir}")
    return output_dir


def main() -> None:
    # An optional argument is convenient for one-off runs; TRAIN_RUN_DIR remains
    # the default so changing the variable above is sufficient.
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_RUN_DIR
    summarize(run_dir)


if __name__ == "__main__":
    main()
