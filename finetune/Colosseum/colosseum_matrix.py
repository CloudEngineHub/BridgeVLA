#!/usr/bin/env python3
"""task x variation success-rate matrix + the paper-protocol overall average for one Colosseum experiment.

Reads every eval_results_*.csv under one eval experiment directory (RESULTS_DIR), aggregates them into
a task x setting success-rate matrix, and computes the COLOSSEUM average **the way the original
BridgeVLA paper does (Tab. colosseum)**. The key points (why it is not just averaging every cell):

  * variation 0 = no_variations = the in-distribution "clean baseline" (the Original column of the
    paper's per-task table). It is reference only and **excluded** from the COLOSSEUM average —
    otherwise this easiest column would inflate it (neither the original 64.0% nor any headline in this repo includes it).
  * variations 1..14 = the paper's 14 generalization settings:
    all_mixed(1) + 12 individual perturbations (2..12, 14) + rlbench_variations(13).
  * COLOSSEUM average = each of those 14 settings is first averaged across tasks (its column mean),
    then those column means are averaged with **equal weight** (mean of column means). That is the
    "average success rate of all evaluated tasks for every perturbation" from appendix B, averaged over
    settings — equally weighted per setting, not per cell/episode (the original 64.0% is exactly the mean of its 14 column means).
  * variations >=15 (rlbench_and_colosseum / friction / mass) are not evaluated by the original; even if
    they appear in the directory they are only annotated in the matrix and **excluded** from the average.

Variation indices and names follow robot-colosseum's data_collection order (see the variation_name list in
robot-colosseum/colosseum/assets/json/<task>.json for each task).

Usage:
    python3 colosseum_matrix.py <RESULTS_DIR>
Output:
    <RESULTS_DIR>/summary_matrix.csv  (columns = Original + the settings present + pert_mean(14)),
    plus a readable matrix and the two overall averages (Original baseline / COLOSSEUM 14-setting) on stdout.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

# spreadsheet idx -> (short column code matching the paper table, the variation_name used in robot-colosseum).
SETTING = {
    0: ("Original", "no_variations"),
    1: ("All-Perturb", "all_mixed"),
    2: ("MO-COLOR", "manip_obj_color"),
    3: ("RO-COLOR", "recv_obj_color"),
    4: ("MO-TEXTURE", "manip_obj_tex"),
    5: ("RO-TEXTURE", "recv_obj_tex"),
    6: ("MO-SIZE", "manip_obj_size"),
    7: ("RO-SIZE", "recv_obj_size"),
    8: ("Light-Color", "light_color"),
    9: ("Table-Color", "table_color"),
    10: ("Table-Texture", "table_texture"),
    11: ("Distractor", "distractor"),
    12: ("Bg-Texture", "background_texture"),
    13: ("RLBench", "rlbench_variations"),
    14: ("Camera-Pose", "camera_pose"),
    15: ("RLBench+COL", "rlbench_and_colosseum"),
    16: ("Friction", "friction"),
    17: ("Mass", "mass"),
}
# The variation indices of the 14 settings that count towards the COLOSSEUM average.
PAPER_SETTINGS = list(range(1, 15))  # 1..14
BASELINE_IDX = 0  # no_variations, reference only, excluded from the average


def load_cells(results_dir: str):
    """(base_task, variation) -> success rate (0-100); reads eval_results_*.csv."""
    paths = sorted(glob.glob(os.path.join(results_dir, "eval_results_*.csv")))
    cells: dict[tuple[str, int], float] = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                task = (row.get("task") or "").strip()
                parts = task.split("_")
                if len(parts) < 2 or not parts[-1].isdigit():
                    continue
                base, var = "_".join(parts[:-1]), int(parts[-1])
                value = (row.get("success rate") or "").strip()
                try:
                    cells[(base, var)] = float(value)
                except ValueError:
                    pass
    return cells, paths


def mean(values) -> float:
    valid = [x for x in values if x == x]  # drop NaN
    return sum(valid) / len(valid) if valid else float("nan")


def fmt(x: float) -> str:
    return f"{x:.1f}" if x == x else "-"  # NaN -> "-"


def code(var: int) -> str:
    return SETTING.get(var, (f"var{var}", f"var{var}"))[0]


def summarize(results_dir: str) -> None:
    cells, paths = load_cells(results_dir)
    if not cells:
        print(f"[Warn] no parsable eval_results_*.csv rows in {results_dir}; skip summary")
        return

    bases = sorted({b for b, _ in cells})
    vars_present = sorted({v for _, v in cells})

    # Column mean per variation (across tasks, counting only tasks that have data).
    col_mean = {
        v: mean([cells[(b, v)] for b in bases if (b, v) in cells])
        for v in vars_present
    }

    reported = [v for v in PAPER_SETTINGS if v in col_mean]
    missing = [v for v in PAPER_SETTINGS if v not in col_mean]
    extra = [v for v in vars_present if v not in PAPER_SETTINGS and v != BASELINE_IDX]
    paper_avg = mean([col_mean[v] for v in reported])
    baseline = col_mean.get(BASELINE_IDX, float("nan"))

    # CSV column order: Original (if present) -> the 1..14 present -> any other indices present (annotated).
    ordered = (
        ([BASELINE_IDX] if BASELINE_IDX in col_mean else [])
        + reported
        + extra
    )

    out = os.path.join(results_dir, "summary_matrix.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["task"] + [f"{code(v)}(v{v})" for v in ordered] + ["pert_mean(14)"]
        )
        for base in bases:
            row = [cells.get((base, v), float("nan")) for v in ordered]
            pert = mean([cells.get((base, v), float("nan")) for v in reported])
            writer.writerow([base] + [fmt(x) for x in row] + [fmt(pert)])
        writer.writerow(
            ["variation_mean"]
            + [fmt(col_mean[v]) for v in ordered]
            + [fmt(paper_avg)]
        )

    # ---- stdout: readable matrix ----
    name_w = max(len(s) for s in bases + ["variation_mean"]) + 2
    col_w = max(12, max(len(code(v)) for v in ordered) + 2)
    print()
    print(
        f"===== COLOSSEUM success-rate matrix "
        f"(%, {len(bases)} tasks x {len(ordered)} settings) ====="
    )
    header = "task".ljust(name_w) + "".join(
        f"{code(v):<{col_w}}" for v in ordered
    ) + "pert_mean"
    print(header)
    for base in bases:
        row = [cells.get((base, v), float("nan")) for v in ordered]
        pert = mean([cells.get((base, v), float("nan")) for v in reported])
        print(
            base.ljust(name_w)
            + "".join(f"{fmt(x):<{col_w}}" for x in row)
            + f"{fmt(pert)}"
        )
    print(
        "variation_mean".ljust(name_w)
        + "".join(f"{fmt(col_mean[v]):<{col_w}}" for v in ordered)
        + f"{fmt(paper_avg)}"
    )

    # ---- stdout: the two overall averages, kept clearly apart ----
    print()
    print("----- headline numbers -----")
    if BASELINE_IDX in col_mean:
        print(
            f"  no_variations (Original, in-distribution baseline): "
            f"{fmt(baseline)}%   [reference, NOT in COLOSSEUM avg]"
        )
    print(
        f"  COLOSSEUM avg (paper protocol, {len(reported)}/14 settings, "
        f"mean of col-means): {fmt(paper_avg)}%"
    )
    if missing:
        print(
            f"  [Warn] partial: {len(missing)} of the 14 reported settings have "
            f"no data: {', '.join(f'{code(v)}(v{v})' for v in missing)}"
        )
    if extra:
        print(
            f"  [Note] extra variations present but NOT counted "
            f"(not evaluated by the original): {', '.join(f'{code(v)}(v{v})' for v in extra)}"
        )
    print(f"\n[Info] matrix saved: {out}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python3 colosseum_matrix.py <RESULTS_DIR>")
        sys.exit(2)
    summarize(sys.argv[1])


if __name__ == "__main__":
    main()
