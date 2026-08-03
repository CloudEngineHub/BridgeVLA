"""Summarize GemBench eval results.

Usage:
    python summarize_eval.py <MODEL_LOG_DIR> [--assets-dir DIR] [--trials N]

MODEL_LOG_DIR = the evaluated checkpoint dir (a released ckpt dir or your own
training run dir — both share the same layout), e.g.
    data/bridgevla_ckpt/bridgevla_plus/gembench

It walks <MODEL_LOG_DIR>/eval/<gembench*>/model_*/seed*/{train,test_l2,test_l3,test_l4}/result.json
(any directory under eval/ whose name starts with "gembench", e.g. gembench, gembench_viz)
and, for each (eval_subdir, model_epoch, seed, split):
  * reads up to `trials` (default 20) episodes per taskvar in the order given by
    assets/taskvars_<split>.json,
  * computes per-taskvar success rate (mean of `success` field),
  * writes a per-split detailed txt with every taskvar (base task + its variants),
  * writes a per-seed summary txt aggregating the four splits.

It finally writes, under each <MODEL_LOG_DIR>/eval/<gembench*>/:
  * summary.csv  -- one row per (model_epoch, seed), columns: train / test_l2 / test_l3 / test_l4 / avg
  * summary.txt  -- the same, prettified for reading.

The script is non-destructive and idempotent: existing per-split result.json files are
only read, never modified.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "test_l2", "test_l3", "test_l4")
TASK_JSON_NAMES = {
    "train": "taskvars_train.json",
    "test_l2": "taskvars_test_l2.json",
    "test_l3": "taskvars_test_l3.json",
    "test_l4": "taskvars_test_l4.json",
}
EXPECTED_TASKVARS = {"train": 31, "test_l2": 28, "test_l3": 21, "test_l4": 12}


def load_taskvars(assets_dir: Path, split: str) -> list[str]:
    with open(assets_dir / TASK_JSON_NAMES[split]) as f:
        return json.load(f)


def load_result_lines(result_path: Path) -> list[dict]:
    """Load JSONL. Non-dict / malformed entries are replaced with {} so that
    index-based alignment with the taskvars list is preserved. We warn once per
    file so that stale/corrupt lines are surfaced."""
    out = []
    bad = 0
    with open(result_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = None
            if not isinstance(obj, dict):
                bad += 1
                out.append({})
            else:
                out.append(obj)
    if bad:
        print(f"[Warn] {result_path}: {bad} malformed line(s) replaced with empty records", file=sys.stderr)
    return out


def per_taskvar_rates(results: list[dict], taskvars: list[str], trials: int) -> list[tuple[str, float, int]]:
    """Return [(taskvar, success_rate, n_done), ...] in taskvars order.

    If fewer than `trials` episodes are present for a taskvar, success rate is
    computed over the episodes that *are* there (still divided by n_done, not by
    `trials`) and n_done reports how many trials were recorded.
    """
    out = []
    for i, tv in enumerate(taskvars):
        chunk = results[i * trials : (i + 1) * trials]
        if not chunk:
            out.append((tv, float("nan"), 0))
            continue
        succ = sum(1 for r in chunk if r.get("success") == 1.0)
        out.append((tv, succ / len(chunk), len(chunk)))
    return out


def group_by_task(rows: list[tuple[str, float, int]]) -> dict[str, list[tuple[str, float, int]]]:
    """Group (taskvar, rate, n) rows by their base task name (before '+')."""
    groups: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for tv, rate, n in rows:
        base = tv.split("+", 1)[0]
        groups[base].append((tv, rate, n))
    return groups


def write_split_detail(path: Path, split: str, rows: list[tuple[str, float, int]], trials: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_task(rows)
    total_rate = [r for _, r, _ in rows if r == r]  # drop NaN
    overall = sum(total_rate) / len(total_rate) if total_rate else float("nan")
    missing = [tv for tv, _, n in rows if n < trials]

    with open(path, "w") as f:
        f.write(f"[split={split}]  taskvars={len(rows)}  trials_per_taskvar={trials}\n")
        f.write(f"overall_success_rate = {overall * 100:.2f}% (mean over taskvars)\n")
        if missing:
            f.write(f"incomplete taskvars ({len(missing)}): {', '.join(missing)}\n")
        f.write("\n")
        f.write(f"{'task':<40s}{'variant':<12s}{'SR':>8s}{'trials':>10s}\n")
        f.write("-" * 70 + "\n")
        for base in sorted(groups):
            variants = groups[base]
            if len(variants) == 1:
                tv, rate, n = variants[0]
                var = tv.split("+", 1)[1] if "+" in tv else "-"
                rate_s = f"{rate * 100:.2f}%" if rate == rate else "N/A"
                f.write(f"{base:<40s}{'+' + var:<12s}{rate_s:>8s}{n:>10d}\n")
            else:
                vrates = [r for _, r, _ in variants if r == r]
                task_mean = sum(vrates) / len(vrates) if vrates else float("nan")
                mean_s = f"{task_mean * 100:.2f}%" if task_mean == task_mean else "N/A"
                f.write(f"{base:<40s}{'(avg)':<12s}{mean_s:>8s}{'':>10s}\n")
                for tv, rate, n in variants:
                    var = tv.split("+", 1)[1] if "+" in tv else "-"
                    rate_s = f"{rate * 100:.2f}%" if rate == rate else "N/A"
                    f.write(f"{'':<40s}{'+' + var:<12s}{rate_s:>8s}{n:>10d}\n")
        f.write("\n")


def write_seed_summary(path: Path, per_split_overall: dict[str, float], per_split_counts: dict[str, tuple[int, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{'split':<10s}{'overall_SR':>14s}{'taskvars':>12s}{'expected':>10s}\n")
        f.write("-" * 46 + "\n")
        for split in SPLITS:
            if split in per_split_overall:
                rate = per_split_overall[split]
                done, exp = per_split_counts[split]
                f.write(f"{split:<10s}{rate * 100:>13.2f}%{done:>12d}{exp:>10d}\n")
            else:
                f.write(f"{split:<10s}{'MISSING':>14s}{'-':>12s}{EXPECTED_TASKVARS[split]:>10d}\n")
        valid = [v for v in per_split_overall.values() if v == v]
        avg = sum(valid) / len(valid) if valid else float("nan")
        f.write("-" * 46 + "\n")
        avg_s = f"{avg * 100:.2f}%" if avg == avg else "N/A"
        f.write(f"{'avg':<10s}{avg_s:>14s}\n")


_MODEL_RE = re.compile(r"^model_(\d+)$")
_SEED_RE = re.compile(r"^seed(\d+)$")


def discover_eval_roots(root: Path) -> list[Path]:
    """Return all <root>/eval/gembench* directories (e.g. gembench, gembench_viz)."""
    eval_dir = root / "eval"
    if not eval_dir.is_dir():
        raise SystemExit(f"[Error] expected eval directory not found: {eval_dir}")
    out = [p for p in sorted(eval_dir.iterdir()) if p.is_dir() and p.name.startswith("gembench")]
    if not out:
        raise SystemExit(f"[Error] no gembench* subdirectories under {eval_dir}")
    return out


def discover(gem_root: Path):
    """Yield (model_epoch:int, seed:int, model_dir:Path, seed_dir:Path) tuples
    for a given eval subdirectory (e.g. .../eval/gembench or .../eval/gembench_viz)."""
    if not gem_root.is_dir():
        raise SystemExit(f"[Error] expected eval directory not found: {gem_root}")
    for model_dir in sorted(gem_root.iterdir()):
        m = _MODEL_RE.match(model_dir.name)
        if not m or not model_dir.is_dir():
            continue
        epoch = int(m.group(1))
        for seed_dir in sorted(model_dir.iterdir()):
            s = _SEED_RE.match(seed_dir.name)
            if not s or not seed_dir.is_dir():
                continue
            yield epoch, int(s.group(1)), model_dir, seed_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_log_dir", type=Path, help="e.g. .../train_gembench/<run_name>")
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
        help="Directory containing taskvars_{train,test_l2,test_l3,test_l4}.json",
    )
    ap.add_argument("--trials", type=int, default=20, help="Trials per taskvar (default 20)")
    args = ap.parse_args()

    root: Path = args.model_log_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"[Error] not a directory: {root}")

    assets_dir: Path = args.assets_dir
    taskvars_by_split = {s: load_taskvars(assets_dir, s) for s in SPLITS}

    eval_roots = discover_eval_roots(root)
    any_written = False

    for gem_root in eval_roots:
        summary_rows = []  # one per (epoch, seed) -- for the compact overview table
        all_runs = []  # one per (epoch, seed) -- carries per-split per-taskvar rows for the hierarchical view

        for epoch, seed, model_dir, seed_dir in discover(gem_root):
            per_split_overall: dict[str, float] = {}
            per_split_counts: dict[str, tuple[int, int]] = {}
            per_split_rows: dict[str, list[tuple[str, float, int]]] = {}
            for split in SPLITS:
                result_path = seed_dir / split / "result.json"
                if not result_path.is_file():
                    continue
                taskvars = taskvars_by_split[split]
                results = load_result_lines(result_path)
                rows = per_taskvar_rates(results, taskvars, args.trials)
                valid_rates = [r for _, r, _ in rows if r == r]
                overall = sum(valid_rates) / len(valid_rates) if valid_rates else float("nan")
                per_split_overall[split] = overall
                per_split_counts[split] = (sum(1 for _, _, n in rows if n > 0), len(taskvars))
                per_split_rows[split] = rows
                detail_path = seed_dir / split / "result_detail.txt"
                write_split_detail(detail_path, split, rows, args.trials)

            if per_split_overall:
                seed_summary_path = seed_dir / "summary.txt"
                write_seed_summary(seed_summary_path, per_split_overall, per_split_counts)
                valid = [v for v in per_split_overall.values() if v == v]
                avg = sum(valid) / len(valid) if valid else float("nan")
                summary_rows.append({
                    "model_epoch": epoch,
                    "seed": seed,
                    "train": per_split_overall.get("train", float("nan")),
                    "test_l2": per_split_overall.get("test_l2", float("nan")),
                    "test_l3": per_split_overall.get("test_l3", float("nan")),
                    "test_l4": per_split_overall.get("test_l4", float("nan")),
                    "avg": avg,
                })
                all_runs.append({
                    "epoch": epoch,
                    "seed": seed,
                    "per_split_overall": per_split_overall,
                    "per_split_counts": per_split_counts,
                    "per_split_rows": per_split_rows,
                    "avg": avg,
                })

        if not summary_rows:
            print(f"[Warn] no result.json files found under {gem_root}")
            continue

        summary_rows.sort(key=lambda r: (r["model_epoch"], r["seed"]))

        csv_path = gem_root / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_epoch", "seed", "train", "test_l2", "test_l3", "test_l4", "avg"])
            for r in summary_rows:
                w.writerow([
                    r["model_epoch"],
                    r["seed"],
                    f"{r['train']:.4f}" if r["train"] == r["train"] else "",
                    f"{r['test_l2']:.4f}" if r["test_l2"] == r["test_l2"] else "",
                    f"{r['test_l3']:.4f}" if r["test_l3"] == r["test_l3"] else "",
                    f"{r['test_l4']:.4f}" if r["test_l4"] == r["test_l4"] else "",
                    f"{r['avg']:.4f}" if r["avg"] == r["avg"] else "",
                ])

        def fmt(v):
            return f"{v * 100:6.2f}%" if v == v else "   N/A"

        # ---- Section 1: compact overview table ----
        overview_header = f"{'epoch':>6s}{'seed':>6s}{'train':>10s}{'test_l2':>10s}{'test_l3':>10s}{'test_l4':>10s}{'avg':>10s}"
        overview_title = f"## Overview [{gem_root.name}] (epoch x level)"
        overview_lines = [overview_title, "", overview_header, "-" * len(overview_header)]
        for r in summary_rows:
            overview_lines.append(
                f"{r['model_epoch']:>6d}{r['seed']:>6d}{fmt(r['train']):>10s}{fmt(r['test_l2']):>10s}{fmt(r['test_l3']):>10s}{fmt(r['test_l4']):>10s}{fmt(r['avg']):>10s}"
            )
        overview = "\n".join(overview_lines) + "\n"

        # ---- Section 2: hierarchical detail — epoch -> level -> task -> variant ----
        detail_lines: list[str] = []
        for run in all_runs:
            epoch = run["epoch"]
            seed = run["seed"]
            detail_lines.append("")
            detail_lines.append("=" * 78)
            detail_lines.append(f"# [Epoch {epoch} | seed {seed}]   avg = {fmt(run['avg']).strip()}")
            detail_lines.append("=" * 78)
            for split in SPLITS:
                if split not in run["per_split_overall"]:
                    detail_lines.append(f"\n## {split}  — MISSING (not evaluated yet)")
                    continue
                overall = run["per_split_overall"][split]
                done, exp = run["per_split_counts"][split]
                tag = "" if done == exp else f"  [INCOMPLETE {done}/{exp} taskvars]"
                detail_lines.append(f"\n## {split}  overall = {fmt(overall).strip()}  ({done}/{exp} taskvars){tag}")
                rows = run["per_split_rows"][split]
                groups = group_by_task(rows)
                for base in sorted(groups):
                    variants = groups[base]
                    if len(variants) == 1:
                        tv, rate, n = variants[0]
                        var = tv.split("+", 1)[1] if "+" in tv else "-"
                        rate_s = f"{rate * 100:6.2f}%" if rate == rate else "   N/A"
                        detail_lines.append(f"### {base}")
                        detail_lines.append(f"      +{var:<6s}  SR = {rate_s}   ({n}/{args.trials} trials)")
                    else:
                        vrates = [r for _, r, _ in variants if r == r]
                        task_mean = sum(vrates) / len(vrates) if vrates else float("nan")
                        mean_s = f"{task_mean * 100:6.2f}%" if task_mean == task_mean else "   N/A"
                        detail_lines.append(f"### {base}   (avg over {len(variants)} variants = {mean_s.strip()})")
                        for tv, rate, n in variants:
                            var = tv.split("+", 1)[1] if "+" in tv else "-"
                            rate_s = f"{rate * 100:6.2f}%" if rate == rate else "   N/A"
                            detail_lines.append(f"      +{var:<6s}  SR = {rate_s}   ({n}/{args.trials} trials)")

        txt_path = gem_root / "summary.txt"
        full_txt = overview + "\n".join(detail_lines) + "\n"
        with open(txt_path, "w") as f:
            f.write(full_txt)

        # Console output: print the compact overview only; the hierarchy lives in the file.
        print(overview, end="")
        print(f"[OK] wrote {csv_path}")
        print(f"[OK] wrote {txt_path}  (overview + epoch/level/task hierarchy)\n")
        any_written = True

    if not any_written:
        print("[Warn] no summaries written — no result.json files found in any eval/gembench* subdirectory.")


if __name__ == "__main__":
    main()
