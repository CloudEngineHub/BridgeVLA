#!/usr/bin/env python3
"""Summarize RLBench success rates for every model and eval timestamp.

The output contains one CSV per ``model_*`` directory. Tasks are rows, eval
timestamps are columns, and the final row is the average success rate for each
timestamp. Timestamp directories without a result CSV are retained as empty
columns.
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
# can be summarised directly; point RLBENCH_SUMMARY_DIR at one instead of editing this file.
TRAIN_RUN_DIR = Path(
    os.environ.get("RLBENCH_SUMMARY_DIR")
    or (os.environ.get("BRIDGEVLA_RELEASE_CKPT_DIR", "data/bridgevla_ckpt/bridgevla_plus") + "/rlbench")
)


RLBENCH_TASKS = [
    "close_jar",
    "reach_and_drag",
    "insert_onto_square_peg",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "light_bulb_in",
    "slide_block_to_color_target",
    "place_shape_in_shape_sorter",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

MODEL_DIR_RE = re.compile(r"^model_(\d+)(?:_(.*))?$")
# The eval directory name = eval.sh's EVAL_LOG_NAME, a bare timestamp by default; a descriptive suffix
# may follow (e.g. 20260730_143000_hold25) so several ablations of one ckpt are self-explanatory in the summary table.
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d{6}(?:_[\w.+-]+)?$")


def model_sort_key(path: Path) -> tuple[int, str]:
    match = MODEL_DIR_RE.fullmatch(path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), match.group(2) or "")


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


def ordered_tasks(results_by_timestamp: Dict[str, Dict[str, float]]) -> List[str]:
    observed = {
        task
        for timestamp_results in results_by_timestamp.values()
        for task in timestamp_results
    }
    extras = sorted(observed.difference(RLBENCH_TASKS))
    return [*RLBENCH_TASKS, *extras]


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


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

        writer.writerow(
            [
                "Average",
                *[
                    "" if value is None else round(value, 2)
                    for value in (
                        average(results_by_timestamp[timestamp].values())
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
        raise FileNotFoundError(f"RLBench eval directory does not exist: {eval_dir}")

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

    output_dir = eval_dir / "rlbench_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for model_dir in model_dirs:
        timestamps, results_by_timestamp, empty_results = collect_model_results(
            model_dir
        )
        if not timestamps:
            print(f"Skip {model_dir.name}: no timestamp directories")
            continue

        output_path = output_dir / f"{model_dir.name}_summary.csv"
        task_count = write_model_csv(output_path, timestamps, results_by_timestamp)
        generated += 1
        print(
            f"{model_dir.name}: {len(timestamps)} timestamps, "
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
