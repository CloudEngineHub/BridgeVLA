#!/usr/bin/env python3
"""
Build a per-variation symlink tree alongside the released MemoryBench data so
client.py can invoke task.reset_to_demo() with deterministic episode ordering.

Released layout (sam2act / RVT convention):
  <root>/<task>/all_variations/episodes/episode<i>/
      ├── variation_number.pkl    # 0-based variation id
      ├── variation_descriptions.pkl
      ├── low_dim_obs.pkl
      └── front_rgb / ... / wrist_depth / ...

RLBench's Environment.get_demos() expects:
  <root>/<task>/variation<vid>/episodes/episode<i>/

We materialise the second view as symlinks. Source remains untouched.

Usage (the ${BRIDGEVLA_DATA_ROOT} path prefix is exported by the launcher .sh scripts, or set it yourself):
  python reorg_data_for_eval.py \
      --src "${BRIDGEVLA_DATA_ROOT}/memorybench/data/test" \
      --dst "${BRIDGEVLA_DATA_ROOT}/memorybench/eval_layout/test"
"""
import argparse
import os
import pickle
import shutil


def main(src_root: str, dst_root: str, force: bool):
    if not os.path.isdir(src_root):
        raise SystemExit(f"src not found: {src_root}")
    if force and os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root, exist_ok=True)

    for task in sorted(os.listdir(src_root)):
        src_av = os.path.join(src_root, task, "all_variations", "episodes")
        if not os.path.isdir(src_av):
            continue
        per_var = {}  # variation -> [(new_idx, src_episode_dir)]
        eps = sorted(os.listdir(src_av), key=lambda n: int(n.replace("episode", "")))
        for ep in eps:
            ep_dir = os.path.join(src_av, ep)
            vn_path = os.path.join(ep_dir, "variation_number.pkl")
            if not os.path.exists(vn_path):
                continue
            with open(vn_path, "rb") as f:
                vid = int(pickle.load(f))
            per_var.setdefault(vid, []).append(ep_dir)
        for vid, ep_dirs in per_var.items():
            target = os.path.join(dst_root, task, f"variation{vid}", "episodes")
            os.makedirs(target, exist_ok=True)
            for new_idx, src_ep in enumerate(ep_dirs):
                link_path = os.path.join(target, f"episode{new_idx}")
                if os.path.exists(link_path) or os.path.islink(link_path):
                    continue
                os.symlink(os.path.abspath(src_ep), link_path)
            print(f"[reorg] {task} variation{vid}: {len(ep_dirs)} episodes -> {target}")
    print(f"[reorg] DONE. Per-variation tree at {dst_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Memorybench root containing <task>/all_variations/episodes/")
    parser.add_argument("--dst", required=True,
                        help="Output dir to populate with <task>/variation<vid>/episodes/ symlinks")
    parser.add_argument("--force", action="store_true",
                        help="Wipe --dst first if it exists.")
    args = parser.parse_args()
    main(args.src, args.dst, args.force)
