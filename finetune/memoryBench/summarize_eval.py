"""Summarize MemoryBench eval results.

Usage:
    python summarize_eval.py <MODEL_LOG_DIR> [--assets-dir DIR] [--trials N]

MODEL_LOG_DIR = the evaluated checkpoint dir (a released ckpt dir or your own
training run dir — both share the same layout), e.g.
    data/bridgevla_ckpt/bridgevla_plus/memorybench

It walks <MODEL_LOG_DIR>/eval/<memorybench*>/model_*/seed*/result.jsonl
(any directory under eval/ whose name starts with "memorybench", e.g. memorybench,
memorybench_viz) and, for each (eval_subdir, model_epoch, seed):
  * loads all episode records from result.jsonl (client.py appends one line per
    episode with {episode_id, instr, success, error, nsteps});
  * splits them into per-taskvar chunks by detecting `episode_id` resets to 0
    (client.py is invoked once per taskvar with episode_id in [0, num_episodes));
  * identifies each chunk's base task by matching the first record's `instr`
    against assets/taskvars_instructions.json (exact / substring / token-overlap);
  * within a base task, chunks are mapped in order to the canonical variant
    order from assets/taskvars.json (which is the same order run_client.sh
    iterates taskvars);
  * computes per-taskvar success rate (mean of `success` field);
  * writes a per-seed result_detail.txt with every taskvar (base task + variants);
  * writes a per-seed summary.txt aggregating the base tasks.

It finally writes, under each <MODEL_LOG_DIR>/eval/<memorybench*>/:
  * summary.csv  -- one row per (model_epoch, seed), columns: per base task + overall;
  * summary.txt  -- same compact overview plus an epoch/base_task/variant hierarchy.

The script is non-destructive and idempotent: existing result.jsonl files are only
read, never modified.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_MODEL_RE = re.compile(r"^model_(\d+)(?:_.+)?$")  # model_99 or model_99_05_24_00_54
_SEED_RE = re.compile(r"^seed(\d+)$")
_WORD_RE = re.compile(r"[a-zA-Z]+")

# Discriminative substrings used to map an evaluation instruction (returned by
# RLBench's `task.reset()`) back to its base task. These are kept here (not in
# `assets/taskvars_instructions.json`) because that asset file is also consumed
# by `train.py` for instruction sampling and must stay aligned with the
# instructions the training pipeline expects to see. The hints below only need
# to be distinctive enough to disambiguate the few base tasks in MemoryBench.
_BASE_TASK_HINTS = {
    "put_block_back": ["color patch", "where it was", "put it back", "original color"],
    "rearrange_block": ["rearrange", "rearrangement", "not on the patch", "empty patch"],
    "reopen_drawer": ["drawer"],
}


def load_taskvars(assets_dir: Path) -> list:
    with open(assets_dir / "taskvars.json") as f:
        return json.load(f)


def load_instructions(assets_dir: Path) -> dict:
    with open(assets_dir / "taskvars_instructions.json") as f:
        return json.load(f)


def match_base_task(instr: str, instructions: dict) -> str:
    """Map an instruction string back to its base task name.

    Order of attempts:
      1. exact / substring match against `assets/taskvars_instructions.json`;
      2. substring match against `_BASE_TASK_HINTS` (covers RLBench-returned
         instructions whose phrasing differs from the asset file);
      3. best Jaccard token overlap against the asset phrases (>=0.2).
    Returns None if nothing scores high enough.
    """
    if not instr:
        return None
    s = instr.strip().lower()
    # 1a. exact match against asset
    for base, phrases in instructions.items():
        for p in phrases:
            if p.strip().lower() == s:
                return base
    # 1b. substring match against asset (either direction)
    for base, phrases in instructions.items():
        for p in phrases:
            pl = p.strip().lower()
            if pl and (pl in s or s in pl):
                return base
    # 2. distinctive hints (handles the common case where RLBench returns
    # phrasings that don't appear verbatim in the asset file)
    for base, hints in _BASE_TASK_HINTS.items():
        for h in hints:
            if h.lower() in s:
                return base
    # 3. best token overlap (Jaccard) against asset phrases
    s_tokens = set(_WORD_RE.findall(s))
    best_base, best_score = None, 0.0
    for base, phrases in instructions.items():
        for p in phrases:
            p_tokens = set(_WORD_RE.findall(p.lower()))
            if not p_tokens:
                continue
            inter = s_tokens & p_tokens
            union = s_tokens | p_tokens
            score = len(inter) / len(union) if union else 0.0
            if score > best_score:
                best_score, best_base = score, base
    return best_base if best_score >= 0.2 else None


def load_records(result_path: Path) -> list:
    """Load JSONL. Malformed lines are skipped with a single warning per file."""
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
                continue
            out.append(obj)
    if bad:
        print(f"[Warn] {result_path}: {bad} malformed line(s) skipped", file=sys.stderr)
    return out


def chunk_by_episode_reset(records: list) -> list:
    """Split records into chunks at every `episode_id == 0` (after the first one).

    client.py rolls out one taskvar per invocation with episode_id in [0, N);
    re-running for a new taskvar appends to the same jsonl, so a reset to 0
    marks the boundary between taskvars.
    """
    chunks = []
    cur = []
    for rec in records:
        eid = rec.get("episode_id")
        if isinstance(eid, int) and eid == 0 and cur:
            chunks.append(cur)
            cur = []
        cur.append(rec)
    if cur:
        chunks.append(cur)
    return chunks


def label_chunks(chunks: list, taskvars: list, instructions: dict) -> list:
    """Return [(taskvar_name, chunk), ...] in the input order.

    For each chunk we identify its base task by matching the chunk's first
    instruction; within a base task, chunks are then consumed in the canonical
    variant order from `taskvars.json` (the same order `run_client.sh`
    iterates). If a base task's chunk count exceeds its variant count we cycle
    via modulo so that repeated rollouts (e.g. the same `result.jsonl` written
    by multiple `run_client.sh` invocations) get aggregated into the right
    taskvar by the downstream `merge_runs` step.
    """
    variants_by_base = defaultdict(list)
    for tv in taskvars:
        base = tv.split("+", 1)[0]
        variants_by_base[base].append(tv)

    cursor = defaultdict(int)
    out = []
    for chunk in chunks:
        if not chunk:
            continue
        instr = (chunk[0].get("instr") or "").strip()
        base = match_base_task(instr, instructions)
        if base is None or base not in variants_by_base:
            print(f"[Warn] could not identify base task for chunk starting with "
                  f"instr={instr!r}", file=sys.stderr)
            out.append((f"unknown_{len(out)}", chunk))
            continue
        variants = variants_by_base[base]
        idx = cursor[base] % len(variants)
        cursor[base] += 1
        out.append((variants[idx], chunk))

    # Sanity check: warn if any base task's chunk count is not an exact multiple
    # of its variant count (suggests a partial rollout that may be misaligned).
    for base, count in cursor.items():
        n_var = len(variants_by_base.get(base, []))
        if n_var and count % n_var != 0:
            print(f"[Warn] {base!r}: {count} chunk(s) is not a multiple of "
                  f"{n_var} variants -- variant assignment may be misaligned "
                  f"(likely a partial rollout)", file=sys.stderr)
    return out


def merge_runs(labelled: list, taskvars: list):
    """Pool episodes for taskvars that appear multiple times (re-runs).

    Returns
    -------
    rows : list of (taskvar, success_rate, n_episodes) in canonical
        `taskvars.json` order. Unrecognised taskvar names are appended at the
        end so nothing is silently dropped.
    runs_per_taskvar : dict[str, int]
        Number of chunks (i.e. distinct `client.py` invocations) that
        contributed to each taskvar. Values > 1 mean the taskvar was
        re-rolled-out and its episodes have been pooled.
    """
    merged = defaultdict(list)
    runs_per_taskvar = defaultdict(int)
    for tv, chunk in labelled:
        merged[tv].extend(chunk)
        runs_per_taskvar[tv] += 1

    rows = []
    seen = set()
    for tv in taskvars:
        eps = merged.get(tv, [])
        seen.add(tv)
        if not eps:
            rows.append((tv, float("nan"), 0))
            continue
        succ = sum(1 for r in eps if float(r.get("success", 0.0)) >= 1.0)
        rows.append((tv, succ / len(eps), len(eps)))
    # surface unknown labels (failed matches, etc.) so the user can spot them
    for tv, eps in merged.items():
        if tv in seen or not eps:
            continue
        succ = sum(1 for r in eps if float(r.get("success", 0.0)) >= 1.0)
        rows.append((tv, succ / len(eps), len(eps)))
    return rows, dict(runs_per_taskvar)


def format_duplicate_banner(runs_per_taskvar: dict) -> str:
    """Build a prominent banner listing any taskvars that were rolled out
    more than once (and whose episodes have therefore been pooled). Returns
    an empty string if there are no duplicates."""
    dups = sorted(
        ((tv, n) for tv, n in runs_per_taskvar.items() if n > 1),
        key=lambda x: x[0],
    )
    if not dups:
        return ""
    lines = [
        "!" * 78,
        "!! DUPLICATE ROLLOUTS DETECTED -- episodes have been POOLED per taskvar",
        "!! (each taskvar's success rate below is computed over the union of",
        "!!  episodes from every re-run found in result.jsonl)",
        "!!",
    ]
    width = max(len(tv) for tv, _ in dups)
    for tv, n in dups:
        lines.append(f"!!   {tv:<{width}s}  x {n} runs")
    lines.append("!" * 78)
    lines.append("")
    return "\n".join(lines) + "\n"


def group_by_base(rows: list) -> dict:
    groups = defaultdict(list)
    for tv, r, n in rows:
        base = tv.split("+", 1)[0]
        groups[base].append((tv, r, n))
    return groups


def write_seed_detail(path: Path, rows: list, trials: int, dup_banner: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_base(rows)
    valid = [r for _, r, _ in rows if r == r]
    overall = sum(valid) / len(valid) if valid else float("nan")
    missing = [tv for tv, _, n in rows if n < trials]

    with open(path, "w") as f:
        if dup_banner:
            f.write(dup_banner)
        f.write(f"taskvars={len(rows)}  trials_per_taskvar={trials}\n")
        f.write(f"overall_success_rate = {overall * 100:.2f}% (mean over taskvars)\n")
        if missing:
            f.write(f"incomplete taskvars ({len(missing)}): {', '.join(missing)}\n")
        f.write("\n")
        f.write(f"{'task':<24s}{'variant':<10s}{'SR':>10s}{'trials':>10s}\n")
        f.write("-" * 54 + "\n")
        for base in sorted(groups):
            variants = groups[base]
            if len(variants) == 1:
                tv, r, n = variants[0]
                var = tv.split("+", 1)[1] if "+" in tv else "-"
                rate_s = f"{r * 100:.2f}%" if r == r else "N/A"
                f.write(f"{base:<24s}{'+' + var:<10s}{rate_s:>10s}{n:>10d}\n")
            else:
                vr = [r for _, r, _ in variants if r == r]
                m = sum(vr) / len(vr) if vr else float("nan")
                ms = f"{m * 100:.2f}%" if m == m else "N/A"
                f.write(f"{base:<24s}{'(avg)':<10s}{ms:>10s}{'':>10s}\n")
                for tv, r, n in variants:
                    var = tv.split("+", 1)[1] if "+" in tv else "-"
                    rate_s = f"{r * 100:.2f}%" if r == r else "N/A"
                    f.write(f"{'':<24s}{'+' + var:<10s}{rate_s:>10s}{n:>10d}\n")
        f.write("\n")


def write_seed_summary(path: Path, base_tasks: list, per_base: dict,
                       per_base_counts: dict, overall: float,
                       dup_banner: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if dup_banner:
            f.write(dup_banner)
        f.write(f"{'base_task':<24s}{'SR':>10s}{'variants':>12s}\n")
        f.write("-" * 46 + "\n")
        for base in base_tasks:
            rate = per_base.get(base, float("nan"))
            done, exp = per_base_counts.get(base, (0, 0))
            rate_s = f"{rate * 100:>9.2f}%" if rate == rate else "       N/A"
            f.write(f"{base:<24s}{rate_s:>10s}{f'{done}/{exp}':>12s}\n")
        f.write("-" * 46 + "\n")
        ov_s = f"{overall * 100:>9.2f}%" if overall == overall else "       N/A"
        f.write(f"{'overall':<24s}{ov_s:>10s}\n")


def discover_eval_roots(root: Path) -> list:
    """Return all <root>/eval/memorybench* directories."""
    eval_dir = root / "eval"
    if not eval_dir.is_dir():
        raise SystemExit(f"[Error] expected eval directory not found: {eval_dir}")
    out = [p for p in sorted(eval_dir.iterdir())
           if p.is_dir() and p.name.startswith("memorybench")]
    if not out:
        raise SystemExit(f"[Error] no memorybench* subdirectories under {eval_dir}")
    return out


def discover_runs(mb_root: Path):
    """Yield (epoch, seed, model_dir, seed_dir) tuples under one eval/memorybench* dir."""
    for model_dir in sorted(mb_root.iterdir()):
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model_log_dir", type=Path,
                    help="e.g. .../train_memorybench/<run_name>")
    ap.add_argument(
        "--assets-dir", type=Path,
        default=Path(__file__).resolve().parent / "assets",
        help="Directory containing taskvars.json and taskvars_instructions.json",
    )
    ap.add_argument("--trials", type=int, default=25,
                    help="Episodes per taskvar (default 25, matches client.py default)")
    args = ap.parse_args()

    root: Path = args.model_log_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"[Error] not a directory: {root}")

    assets_dir: Path = args.assets_dir
    taskvars = load_taskvars(assets_dir)
    instructions = load_instructions(assets_dir)
    base_tasks = sorted({tv.split("+", 1)[0] for tv in taskvars})
    variants_by_base = defaultdict(list)
    for tv in taskvars:
        variants_by_base[tv.split("+", 1)[0]].append(tv)

    eval_roots = discover_eval_roots(root)
    any_written = False

    for mb_root in eval_roots:
        summary_rows = []
        all_runs = []

        for epoch, seed, model_dir, seed_dir in discover_runs(mb_root):
            result_path = seed_dir / "result.jsonl"
            if not result_path.is_file():
                continue

            records = load_records(result_path)
            chunks = chunk_by_episode_reset(records)
            labelled = label_chunks(chunks, taskvars, instructions)
            rows, runs_per_taskvar = merge_runs(labelled, taskvars)
            dup_banner = format_duplicate_banner(runs_per_taskvar)
            has_dups = bool(dup_banner)
            if has_dups:
                dup_taskvars = sorted(tv for tv, n in runs_per_taskvar.items() if n > 1)
                print(f"[Warn] {result_path}: duplicate rollouts detected for "
                      f"{len(dup_taskvars)} taskvar(s): {', '.join(dup_taskvars)} "
                      f"-- pooling their episodes", file=sys.stderr)

            # per-base aggregate (mean over the variants that have data)
            groups = group_by_base(rows)
            per_base = {}
            per_base_counts = {}
            for base in base_tasks:
                vs = groups.get(base, [])
                vr = [r for _, r, n in vs if n > 0 and r == r]
                done = sum(1 for _, _, n in vs if n > 0)
                per_base[base] = sum(vr) / len(vr) if vr else float("nan")
                per_base_counts[base] = (done, len(variants_by_base[base]))
            valid = [v for v in per_base.values() if v == v]
            overall = sum(valid) / len(valid) if valid else float("nan")

            write_seed_detail(seed_dir / "result_detail.txt", rows, args.trials,
                              dup_banner=dup_banner)
            write_seed_summary(seed_dir / "summary.txt", base_tasks,
                               per_base, per_base_counts, overall,
                               dup_banner=dup_banner)

            summary_rows.append({
                "model_epoch": epoch, "seed": seed,
                **per_base, "overall": overall,
                "has_dups": has_dups,
            })
            all_runs.append({
                "epoch": epoch, "seed": seed,
                "rows": rows, "per_base": per_base,
                "per_base_counts": per_base_counts, "overall": overall,
                "result_path": result_path,
                "runs_per_taskvar": runs_per_taskvar,
                "has_dups": has_dups,
            })

        if not summary_rows:
            print(f"[Warn] no result.jsonl files found under {mb_root}")
            continue

        summary_rows.sort(key=lambda r: (r["model_epoch"], r["seed"]))

        csv_path = mb_root / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model_epoch", "seed", *base_tasks, "overall"])
            for r in summary_rows:
                w.writerow([
                    r["model_epoch"], r["seed"],
                    *(f"{r[b]:.4f}" if r[b] == r[b] else "" for b in base_tasks),
                    f"{r['overall']:.4f}" if r["overall"] == r["overall"] else "",
                ])

        def fmt(v):
            return f"{v * 100:6.2f}%" if v == v else "   N/A"

        # ---- Aggregated duplicate banner across all (epoch, seed) ----
        dup_rows = [r for r in summary_rows if r.get("has_dups")]
        if dup_rows:
            agg_banner_lines = [
                "!" * 78,
                "!! DUPLICATE ROLLOUTS DETECTED in the following (epoch, seed) entries.",
                "!! Their per-taskvar success rates pool all re-runs found in their",
                "!! result.jsonl. See the corresponding seed dir's result_detail.txt",
                "!! for the exact per-taskvar run counts.",
                "!!",
            ]
            for r in dup_rows:
                agg_banner_lines.append(
                    f"!!   epoch={r['model_epoch']}  seed={r['seed']}"
                )
            agg_banner_lines.append("!" * 78)
            agg_banner_lines.append("")
            agg_dup_banner = "\n".join(agg_banner_lines) + "\n"
        else:
            agg_dup_banner = ""

        # ---- Section 1: compact overview ----
        col_w = max(12, max(len(b) for b in base_tasks) + 2)
        header = (
            f"{'epoch':>6s}{'seed':>6s}"
            + "".join(f"{b:>{col_w}s}" for b in base_tasks)
            + f"{'overall':>10s}"
            + f"{'dup':>6s}"
        )
        overview_lines = [
            f"## Overview [{mb_root.name}] (epoch x base_task)",
            "",
            "(dup column: '*' means this (epoch, seed) had >1 rollout for some",
            " taskvar(s); episodes have been pooled before computing SR)",
            "",
            header,
            "-" * len(header),
        ]
        for r in summary_rows:
            dup_mark = "  *" if r.get("has_dups") else ""
            overview_lines.append(
                f"{r['model_epoch']:>6d}{r['seed']:>6d}"
                + "".join(f"{fmt(r[b]):>{col_w}s}" for b in base_tasks)
                + f"{fmt(r['overall']):>10s}"
                + f"{dup_mark:>6s}"
            )
        overview = "\n".join(overview_lines) + "\n"

        # ---- Section 2: hierarchical detail ----
        detail_lines = []
        for run in all_runs:
            detail_lines.append("")
            detail_lines.append("=" * 78)
            dup_tag = "  [duplicate rollouts pooled]" if run.get("has_dups") else ""
            detail_lines.append(
                f"# [Epoch {run['epoch']} | seed {run['seed']}]   "
                f"overall = {fmt(run['overall']).strip()}{dup_tag}"
            )
            detail_lines.append("=" * 78)
            groups = group_by_base(run["rows"])
            runs_per_tv = run.get("runs_per_taskvar", {})
            for base in base_tasks:
                vr = run["per_base"].get(base, float("nan"))
                done, exp = run["per_base_counts"].get(base, (0, 0))
                vs = groups.get(base, [])
                if not vs:
                    detail_lines.append(f"\n## {base}  - MISSING (0/{exp} variants)")
                    continue
                tag = "" if done == exp else f"  [INCOMPLETE {done}/{exp} variants]"
                detail_lines.append(
                    f"\n## {base}  avg = {fmt(vr).strip()}  ({done}/{exp} variants){tag}"
                )
                for tv, r, n in vs:
                    var = tv.split("+", 1)[1] if "+" in tv else "-"
                    rate_s = f"{r * 100:6.2f}%" if r == r else "   N/A"
                    nr = runs_per_tv.get(tv, 0)
                    dup_suffix = f"  [pooled {nr} runs]" if nr > 1 else ""
                    detail_lines.append(
                        f"      +{var:<6s}  SR = {rate_s}   "
                        f"({n}/{args.trials} trials){dup_suffix}"
                    )

        txt_path = mb_root / "summary.txt"
        with open(txt_path, "w") as f:
            f.write(agg_dup_banner + overview + "\n".join(detail_lines) + "\n")

        # Console output: keep the (potentially long) aggregated banner out of
        # stdout -- the file has it, and per-seed duplicates were already
        # printed via [Warn] earlier. Just print the compact overview.
        print(overview, end="")
        print(f"[OK] wrote {csv_path}")
        print(f"[OK] wrote {txt_path}  (overview + epoch/base_task hierarchy)\n")
        any_written = True

    if not any_written:
        print("[Warn] no summaries written -- no result.jsonl files found in any "
              "eval/memorybench* subdirectory.")


if __name__ == "__main__":
    main()
