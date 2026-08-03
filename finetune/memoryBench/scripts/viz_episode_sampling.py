"""
Walk an entire episode through MemoryBench_Dataset.__getitem__ (exactly the
train-time sampling path: same cache version, same memory bundle, same
keys), and dump one PNG per step that wires up:

    [anchor (frame 0)] [hist_{K-1}] ... [hist_0] [current] [next obs]

Each "column" is the SAME 4-camera union point cloud, pushed through
move_pc_in_bound + place_pc_in_cube (with_mean_or_bounds=False, identical
scene_bounds to training) and rendered by `point_renderer.RVTBoxRenderer`
exactly as MVT.render does at train time. Rows are the three rendered
views (top / front / right).

On TOP of the rendered RGB we overlay the action-target's GT heatmap:
  - GT xyz = sample["gripper_pose"][:3] from the current step (i.e. the
    gripper pose AT cache_slot step_idx+1 = demo frame keyframes[step_idx+1])
  - cube-space projection via the renderer's get_pt_loc_on_img — the
    EXACT same code path agent.get_action_trans uses
  - 2D Gaussian heatmap via mvt_utils.generate_hm_from_pt with the
    sigma stored in exp_cfg (gt_hm_sigma = 1.5), matched to the train loss

so the red blob is the *same red blob* that lights up `mvt1/overlay_*.png`
during training.

Run:
    cd finetune/memoryBench
    bash -c '
      FINETUNE_DIR="$(cd .. && pwd)"
      source "$(conda info --base)/bin/activate" "${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"
      export PYTHONPATH=$FINETUNE_DIR/memoryBench:$FINETUNE_DIR:$FINETUNE_DIR/bridgevla/libs/peract_colab:$FINETUNE_DIR/bridgevla/libs/peract:$FINETUNE_DIR/bridgevla/libs/point-renderer:${PYTHONPATH:-}
      export COPPELIASIM_ROOT=$FINETUNE_DIR/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
      export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}
      python scripts/viz_episode_sampling.py --ep_idx 8 --task reopen_drawer
    '
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from memorybench_dataset import MemoryBench_Dataset
from utils.peract_utils_memorybench import MEMORYBENCH_TASKS, SCENE_BOUNDS

import bridgevla.mvt.utils as mvt_utils
from bridgevla.utils import rvt_utils

from point_renderer.rvt_renderer import RVTBoxRenderer


CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
VIEW_NAMES = ("top", "front", "right")  # MVT three_views order


def _norm_rgb_to_signed(rgb_uint8_chw: np.ndarray) -> torch.Tensor:
    """(3, H, W) uint8 -> (3, H, W) float in [-1, 1]. Matches
    _preprocess_inputs_memorybench's _norm_rgb path.
    """
    x = torch.from_numpy(rgb_uint8_chw.astype(np.float32) / 255.0) * 2.0 - 1.0
    return x


def _frame_to_pc_imgfeat(sample_frame: dict, device: torch.device
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One frame (= a {"rgb": (3,H,W) uint8, "pcd": (3,H,W) float32} per
    camera) -> (pc, img_feat) tensors batched to (1, npts, 3) on `device`.

    Mirrors:
        rgb_signed = (rgb/255)*2 - 1
        pcd        = pcd (already world-frame)
        obs.append([rgb_signed, pcd]); pcds.append(pcd)
        pc, img_feat = get_pc_img_feat(obs, pcds)
        # get_pc_img_feat additionally does (img_feat + 1) / 2 -> [0, 1]
    """
    obs = []
    pcds = []
    for cam in CAMERAS:
        rgb_chw = sample_frame[cam]["rgb"]            # (3, H, W) uint8
        pcd_chw = sample_frame[cam]["pcd"]            # (3, H, W) float32
        rgb_signed = _norm_rgb_to_signed(rgb_chw)
        pcd = torch.from_numpy(pcd_chw.astype(np.float32))
        obs.append([rgb_signed.unsqueeze(0), pcd.unsqueeze(0)])
        pcds.append(pcd.unsqueeze(0))
    pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcds)
    return pc.to(device), img_feat.to(device)


def _render_one_frame(
    renderer: RVTBoxRenderer,
    pc_world: torch.Tensor,        # (1, npts, 3) on device
    img_feat: torch.Tensor,        # (1, npts, 3) on device, [0,1]
    scene_bounds: List[float],
    move_in_bound: bool,
    device: torch.device,
) -> torch.Tensor:
    """Replicates the per-sample subset of MVT.render(... mvt1_or_mvt2=True)
    for our offline pipeline:
      1. move_pc_in_bound (drop points outside scene_bounds; matches
         agent.move_pc_in_bound=True in exp_cfg)
      2. place_pc_in_cube with scene_bounds (place_with_mean=False)
      3. renderer(pc, feat=cat(pc/max_pc, img_feat))  -- add_corr=True,
         norm_corr=True branch, fix_cam=True

    Returns rendered (V, H, W, 3) RGB float in [0, 1].
    """
    if move_in_bound:
        pc_b, img_feat_b = rvt_utils.move_pc_in_bound(
            pc_world, img_feat, scene_bounds, no_op=False,
        )
        pc_one = pc_b[0]
        feat_one = img_feat_b[0]
    else:
        pc_one = pc_world[0]
        feat_one = img_feat[0]

    pc_cube, _ = mvt_utils.place_pc_in_cube(
        pc_one,
        with_mean_or_bounds=False,
        scene_bounds=scene_bounds,
    )

    if pc_cube.shape[0] == 0:
        max_pc = torch.tensor(1.0, device=device)
    else:
        max_pc = torch.max(torch.abs(pc_cube))

    feat_cat = torch.cat([pc_cube / max_pc, feat_one], dim=-1)  # (npts, 6)
    img_out = renderer(pc_cube, feat_cat, fix_cam=True)
    # img_out shape: (V, H, W, 7) — (PC_norm 3) ++ (RGB 3) ++ (depth 1).
    rgb = img_out[..., 3:6].clamp(0, 1)
    return rgb


def _project_wpt_to_views(
    renderer: RVTBoxRenderer,
    wpt_world: torch.Tensor,          # (1, 3) world-frame
    scene_bounds: List[float],
    device: torch.device,
) -> torch.Tensor:
    """Replicates agent.get_action_trans's `wpt_img = mvt.get_pt_loc_on_img`
    leg: pushes the GT world-frame xyz through the cube transform
    (with_mean_or_bounds=False — depends only on scene_bounds, not on the
    per-frame point cloud) and projects to each fix-cam view's pixel.

    Returns (V, 2) pixel coords (one xy per view).
    """
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    center = torch.tensor(
        [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2],
        device=device, dtype=torch.float32,
    )
    scale = 2.0 / max(x_max - x_min, y_max - y_min, z_max - z_min)

    wpt = wpt_world.view(1, 3).to(device).float()
    wpt_cube = (wpt - center) * scale                # (1, 3)

    # renderer expects (bs, np, 3); we use bs=1, np=1.
    wpt_img = renderer.get_pt_loc_on_img(
        wpt_cube.unsqueeze(0), fix_cam=True, dyn_cam_info=None,
    )
    # wpt_img shape: (bs=1, np=1, V, 2) -- per RVTBoxRenderer.
    return wpt_img.squeeze(0).squeeze(0)             # (V, 2)


def _overlay_gt_heatmap(
    rgb_views: torch.Tensor,    # (V, H, W, 3) in [0,1]
    wpt_img: torch.Tensor,      # (V, 2) projected pixel coords
    sigma: float,
    alpha: float = 0.55,
) -> np.ndarray:
    """For each view, draw a Gaussian heatmap (sigma matches gt_hm_sigma)
    centered at wpt_img and alpha-blend onto the rendered RGB.
    Returns (V, H, W, 3) uint8.
    """
    V, H, W, _ = rgb_views.shape
    hm = mvt_utils.generate_hm_from_pt(
        wpt_img.reshape(-1, 2).cpu(), (H, W),
        sigma=sigma, thres_sigma_times=3,
    )  # (V, H, W) — softmax-normalized per-view sums to 1

    hm_np = hm.cpu().numpy()
    # Normalize each view's heatmap so the peak is 1 for visual contrast.
    peaks = hm_np.reshape(V, -1).max(axis=1)
    peaks = np.where(peaks > 1e-8, peaks, 1.0).reshape(V, 1, 1)
    hm_norm = np.clip(hm_np / peaks, 0.0, 1.0)

    rgb_np = (rgb_views.cpu().numpy() * 255.0).astype(np.uint8)
    # Compose RED channel only -- distinguishable from the scene's
    # rendered colors.
    red = np.zeros_like(rgb_np)
    red[..., 0] = 255

    blended = (
        (1.0 - alpha * hm_norm[..., None]) * rgb_np.astype(np.float32)
        + (alpha * hm_norm[..., None]) * red.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _find_dataset_idx(dataset: MemoryBench_Dataset, task: str, ep_idx: int,
                      step_idx: int) -> int:
    for i, entry in enumerate(dataset.index):
        if (entry["task"] == task
                and int(entry["ep_idx"]) == int(ep_idx)
                and int(entry["step_idx"]) == int(step_idx)):
            return i
    raise KeyError(f"({task}, ep={ep_idx}, step={step_idx}) not in dataset.index")


def _episode_steps(dataset: MemoryBench_Dataset, task: str, ep_idx: int) -> List[int]:
    steps = [int(entry["step_idx"]) for entry in dataset.index
             if entry["task"] == task and int(entry["ep_idx"]) == int(ep_idx)]
    return sorted(steps)


def _load_episode_meta(cache_dir: str, task: str, ep_idx: int) -> dict:
    p = os.path.join(cache_dir, task, f"episode{ep_idx}.npz.meta")
    with open(p) as f:
        return json.load(f)


def _build_frame_dict(sample: dict, kind: str, k: int = 0) -> dict:
    """Repack `sample`'s per-cam entries into a {cam: {"rgb", "pcd"}} dict
    indexed by camera name (matching what _frame_to_pc_imgfeat consumes).
    `kind` ∈ {"current", "anchor", "hist", "raw_current"}.
    """
    out = {}
    for cam in CAMERAS:
        if kind == "current":
            sub = sample[cam]
        elif kind == "anchor":
            sub = sample[f"anchor_{cam}"]
        elif kind == "hist":
            sub = sample[f"hist{k}_{cam}"]
        else:
            raise ValueError(kind)
        out[cam] = sub
    return out


def _save_rendered_step_panel(
    sample: dict,
    next_sample: dict | None,
    *,
    renderer: RVTBoxRenderer,
    dataset_idx: int,
    task: str,
    ep_idx: int,
    step_idx: int,
    K: int,
    keyframes: List[int],
    num_keyframes: int,
    scene_bounds: List[float],
    move_in_bound: bool,
    gt_hm_sigma: float,
    device: torch.device,
    save_path: str,
) -> None:
    """Build [anchor, hist_{K-1}..hist_0, current, next] -> 3 rendered views
    each, overlay GT heatmap (=gripper@step_idx+1, projected) on every
    column, save as a (3 rows = views) x (N cols = frames) grid PNG.
    """
    cols_imgs: List[np.ndarray] = []
    col_labels: List[str] = []

    gt_world = torch.tensor(sample["gripper_pose"][:3], dtype=torch.float32).view(1, 3)

    # All columns share the SAME cube transform (place_with_mean=False),
    # so the GT projects to the same pixel on every column. Compute once.
    gt_wpt_img = _project_wpt_to_views(renderer, gt_world, scene_bounds, device)

    # ---- 1) Anchor (cache slot 0, demo frame keyframes[0]) -------------
    anchor_frame = _build_frame_dict(sample, "anchor")
    a_pc, a_feat = _frame_to_pc_imgfeat(anchor_frame, device)
    a_rgb = _render_one_frame(renderer, a_pc, a_feat, scene_bounds,
                              move_in_bound, device)
    a_blended = _overlay_gt_heatmap(a_rgb, gt_wpt_img, gt_hm_sigma)
    cols_imgs.append(a_blended)
    col_labels.append(
        f"anchor\nslot 0  frame {keyframes[0]}\n"
        f"anchor_mask={bool(sample['anchor_mask'])}"
    )

    # ---- 2) History slots (OLDEST first: K-1, K-2, ..., 0) -------------
    # NOTE: ``hist_action`` is no longer emitted by the dataset (the
    # ``action_proj`` MLP was removed from MemoryBlock, so temporal memory
    # is purely visual + per-slot index PE). Labels reflect that.
    hist_mask = sample["hist_mask"]
    for k in reversed(range(K)):
        h_step = step_idx - k - 1
        h_step_clipped = max(h_step, 0)
        valid = bool(hist_mask[k])
        h_frame = _build_frame_dict(sample, "hist", k=k)
        h_pc, h_feat = _frame_to_pc_imgfeat(h_frame, device)
        h_rgb = _render_one_frame(renderer, h_pc, h_feat, scene_bounds,
                                  move_in_bound, device)
        h_blended = _overlay_gt_heatmap(h_rgb, gt_wpt_img, gt_hm_sigma)
        if not valid:
            h_blended = np.zeros_like(h_blended)
        cols_imgs.append(h_blended)
        a_step = step_idx - k
        a_frame = keyframes[a_step] if 0 <= a_step < num_keyframes else "?"
        col_labels.append(
            f"hist k={k}\nslot {h_step} frame {keyframes[h_step_clipped] if valid else '-'}\n"
            f"act@slot {a_step} f{a_frame}\n"
            f"valid={valid}"
        )

    c_frame = _build_frame_dict(sample, "current")
    c_pc, c_feat = _frame_to_pc_imgfeat(c_frame, device)
    c_rgb = _render_one_frame(renderer, c_pc, c_feat, scene_bounds,
                              move_in_bound, device)
    c_blended = _overlay_gt_heatmap(c_rgb, gt_wpt_img, gt_hm_sigma)
    cols_imgs.append(c_blended)
    col_labels.append(
        f"current\nslot {step_idx} frame {keyframes[step_idx]}\n"
        f"GT (red) = action@frame {keyframes[step_idx + 1]}"
    )

    # ---- 4) NEXT obs (= step_idx+1's `current`) ------------------------
    if next_sample is not None:
        n_frame = _build_frame_dict(next_sample, "current")
        n_pc, n_feat = _frame_to_pc_imgfeat(n_frame, device)
        n_rgb = _render_one_frame(renderer, n_pc, n_feat, scene_bounds,
                                  move_in_bound, device)
        # Cube transform is scene-bounds-based, so the same world-frame
        # gt_xyz lands on the SAME pixel here as in `current` -- and that
        # pixel should sit right on the gripper in the next frame.
        n_blended = _overlay_gt_heatmap(n_rgb, gt_wpt_img, gt_hm_sigma)
        cols_imgs.append(n_blended)
        col_labels.append(
            f"NEXT obs\n(= step {step_idx+1}'s current)\n"
            f"slot {step_idx+1} frame {keyframes[step_idx+1]}\n"
            f"red blob = where the gripper just moved TO"
        )
    else:
        nxt_dummy = np.zeros_like(c_blended)
        cols_imgs.append(nxt_dummy)
        col_labels.append(
            f"NEXT obs\n(no dataset entry; last transition)\n"
            f"slot {step_idx+1} frame {keyframes[step_idx+1]}"
        )

    # ---- compose figure ------------------------------------------------
    n_rows = len(VIEW_NAMES)
    n_cols = len(cols_imgs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows),
                             squeeze=False)
    for c, (col_views, lbl) in enumerate(zip(cols_imgs, col_labels)):
        for r in range(n_rows):
            ax = axes[r][c]
            ax.imshow(col_views[r])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lbl, fontsize=8)
            if c == 0:
                ax.set_ylabel(VIEW_NAMES[r], fontsize=10)

    gt_xyz = sample["gripper_pose"][:3]
    gt_quat = sample["gripper_pose"][3:7]
    gt_grip = float(sample["gripper_pose"][7])
    time_feat = float(sample["low_dim_state"][-1])
    suptitle = (
        f"{task} | ep_idx={ep_idx} | step_idx={step_idx}  (dataset_idx={dataset_idx})  "
        f"K={K}  num_keyframes={num_keyframes}\n"
        f"obs@cache_slot {step_idx} (demo frame {keyframes[step_idx]})  "
        f"-> action target @ cache_slot {step_idx+1} (demo frame {keyframes[step_idx+1]})\n"
        f"gt_xyz=[{gt_xyz[0]:+.3f}, {gt_xyz[1]:+.3f}, {gt_xyz[2]:+.3f}]   "
        f"gt_quat=[{gt_quat[0]:+.3f}, {gt_quat[1]:+.3f}, {gt_quat[2]:+.3f}, {gt_quat[3]:+.3f}]   "
        f"gt_grip={gt_grip:.1f}   time_feat={time_feat:+.3f}\n"
        f"red Gaussian = GT projected via renderer.get_pt_loc_on_img (sigma={gt_hm_sigma}, same code path as agent.get_action_trans)"
    )
    fig.suptitle(suptitle, fontsize=9, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Default paths come from the BRIDGEVLA_DATA_ROOT / BRIDGEVLA_LOG_DIR environment variables;
    # when unset they fall back to repo-relative paths.
    _data_root_base = os.environ.get(
        "BRIDGEVLA_DATA_ROOT",
        "data/bridgevla_data",
    )
    _log_dir_base = os.environ.get("BRIDGEVLA_LOG_DIR", os.path.join(_data_root_base, "logs"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=os.path.join(_data_root_base, "memorybench/data/train"),
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(_data_root_base, "memorybench/data/_keyframe_cache/size128_v3"),
    )
    parser.add_argument(
        "--mvt_cfg",
        default=os.path.join(
            os.environ.get("BRIDGEVLA_RELEASE_CKPT_DIR",
                           "data/bridgevla_ckpt/bridgevla_plus"),
            "memorybench/mvt_cfg.yaml",
        ),
        help="mvt_cfg.yaml of the run we want to mirror (for img_size / three_views / flip_top_up). "
             "Either the copy in a released ckpt directory or the one in your own training run works.",
    )
    parser.add_argument("--gt_hm_sigma", type=float, default=1.5,
                        help="Matches exp_cfg.rvt.gt_hm_sigma.")
    parser.add_argument("--task", default="reopen_drawer")
    parser.add_argument("--ep_idx", type=int, default=8)
    parser.add_argument("--k_temporal", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument(
        "--out_dir", default=os.path.join(_log_dir_base, "_sampling_debug"),
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    assert args.task in MEMORYBENCH_TASKS

    # ---- Read just the renderer-relevant bits out of mvt_cfg.yaml ------
    import yaml
    with open(args.mvt_cfg) as f:
        mvt_cfg = yaml.safe_load(f)
    img_size = int(mvt_cfg.get("img_size", 224))
    rend_three_views = bool(mvt_cfg.get("rend_three_views", True))
    rend_oblique_views = bool(mvt_cfg.get("rend_oblique_views", False))
    flip_top_up = bool(mvt_cfg.get("flip_top_up", False))
    add_depth = bool(mvt_cfg.get("add_depth", True))
    assert rend_three_views and not rend_oblique_views, (
        "this script assumes three_views (top/front/right) layout"
    )

    device = torch.device(args.device)
    renderer = RVTBoxRenderer(
        img_size=(img_size, img_size),
        three_views=rend_three_views,
        oblique_views=rend_oblique_views,
        radius=0.012,
        with_depth=add_depth,
        flip_top_up=flip_top_up,
        device=str(device),
        strict_input_device=False,
    )

    # ---- Dataset & episode meta ----------------------------------------
    dataset = MemoryBench_Dataset(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        memory_enabled=True,
        memory_k_temporal=args.k_temporal,
        tasks=MEMORYBENCH_TASKS,
        verbose=False,
    )
    print(f"[viz_episode_sampling] dataset len = {len(dataset)}")

    ep_meta = _load_episode_meta(args.cache_dir, args.task, args.ep_idx)
    keyframes = list(ep_meta["keyframes"])
    num_kp = int(ep_meta["num_keyframes"])
    print(f"[viz_episode_sampling] {args.task} ep_idx={args.ep_idx}  "
          f"num_keyframes={num_kp}  keyframes={keyframes}  "
          f"transitions={num_kp - 1}")

    steps = _episode_steps(dataset, args.task, args.ep_idx)
    assert steps == list(range(num_kp - 1))

    out_dir = os.path.join(
        args.out_dir,
        f"{args.task}_ep{args.ep_idx:04d}_K{args.k_temporal}_rendered",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[viz_episode_sampling] writing into {out_dir}")
    print(f"[viz_episode_sampling] renderer img_size={img_size}  "
          f"three_views={rend_three_views}  flip_top_up={flip_top_up}  "
          f"device={device}")

    # Pre-pull all step samples so step k can see step k+1's "current".
    step_to_sample: dict[int, dict] = {}
    step_to_ds_idx: dict[int, int] = {}
    for s in steps:
        ds_idx = _find_dataset_idx(dataset, args.task, args.ep_idx, s)
        step_to_sample[s] = dataset[ds_idx]
        step_to_ds_idx[s] = ds_idx

    summary_lines = [
        f"task={args.task}  ep_idx={args.ep_idx}  num_keyframes={num_kp}  "
        f"K={args.k_temporal}",
        f"keyframes (demo frame ids) = {keyframes}",
        f"lang_goal = {ep_meta['lang_goal']}",
        "",
    ]
    for s in steps:
        ds_idx = step_to_ds_idx[s]
        sample = step_to_sample[s]
        next_sample = step_to_sample.get(s + 1)
        save_path = os.path.join(out_dir, f"step_{s:03d}.png")
        _save_rendered_step_panel(
            sample, next_sample,
            renderer=renderer,
            dataset_idx=ds_idx,
            task=args.task,
            ep_idx=args.ep_idx,
            step_idx=s,
            K=args.k_temporal,
            keyframes=keyframes,
            num_keyframes=num_kp,
            scene_bounds=SCENE_BOUNDS,
            move_in_bound=True,
            gt_hm_sigma=args.gt_hm_sigma,
            device=device,
            save_path=save_path,
        )
        gt_xyz = sample["gripper_pose"][:3].tolist()
        time_feat = float(sample["low_dim_state"][-1])
        line = (
            f"  step {s:02d}  dataset_idx={ds_idx:5d}  "
            f"obs@frame {keyframes[s]:4d}  ->  action@frame {keyframes[s+1]:4d}  "
            f"gt_xyz=[{gt_xyz[0]:+.3f},{gt_xyz[1]:+.3f},{gt_xyz[2]:+.3f}]  "
            f"time_feat={time_feat:+.3f}  "
            f"hist_mask={[bool(x) for x in sample['hist_mask']]}  "
            f"anchor_mask={bool(sample['anchor_mask'])}"
        )
        summary_lines.append(line)
        print(line)

    with open(os.path.join(out_dir, "SUMMARY.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[viz_episode_sampling] done. PNGs in {out_dir}")


if __name__ == "__main__":
    main()
