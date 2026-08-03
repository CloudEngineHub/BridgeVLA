"""Training-time visualization for MemoryBench.

Mirrors ``GemBench/visualize.py`` 1:1 (same agent, same MVT renderer, same
memory-panel logic) — the only deltas are:

* Camera list / preprocess come from :mod:`utils.peract_utils_memorybench`
  instead of the GemBench variant.
* MemoryBench's ``Dataset.index`` keys episodes by ``ep_idx`` (int) rather
  than ``ep_key`` (str). We pass ``episode_field="ep_idx"`` and format the
  int into a directory name when laying out the per-step viz tree.

Output layout::

    {log_dir}/viz/epoch_{epoch:04d}/
        {task}/episode{ep_idx:04d}/step_{k:03d}/{stage}/...
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from bridgevla.mvt.augmentation import apply_se3_aug_con
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.utils.viz_utils import (
    _save_stage,
    _write_meta,
    group_dataset_by_episode,
    pick_one_episode_per_task,
    stitch_episode_overlays,
)

from utils.peract_utils_memorybench import (
    CAMERAS as MEMORYBENCH_CAMERAS,
    _preprocess_inputs_memorybench,
)


# PaliGemma bf16 attention kernel FPEs on H20 with bs < 4. Pad the forward
# batch up to this minimum by duplicating the first sample; duplicates are
# never written to disk.
MIN_FORWARD_BS = 4


# tensor/dict helpers

def _move_to_device(x, device):
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(a))
    if t.dtype == torch.float64:
        t = t.float()
    return t


def _collate_samples(samples: List[dict]) -> dict:
    """Stack a list of MemoryBench_Dataset samples into a batch dict."""
    assert len(samples) > 0
    keys = samples[0].keys()
    out: dict = {}
    for k in keys:
        v0 = samples[0][k]
        if isinstance(v0, dict):
            out[k] = {
                sub_k: _to_tensor(np.stack([s[k][sub_k] for s in samples], axis=0))
                for sub_k in v0
            }
        elif isinstance(v0, np.ndarray):
            out[k] = _to_tensor(np.stack([s[k] for s in samples], axis=0))
        elif isinstance(v0, (int, float, np.floating, np.integer)):
            out[k] = torch.tensor([float(s[k]) for s in samples])
        else:
            out[k] = [s[k] for s in samples]
    return out


# forward pass

@torch.no_grad()
def _run_viz_forward(
    agent,
    samples: List[dict],
    cameras: Sequence[str],
    apply_aug: bool = False,
) -> Tuple[dict, torch.Tensor, torch.Tensor, int, int, int]:
    """Run a single eval-mode forward through ``agent._net_mod`` and return
    the tensors the visualization layer consumes.
    """
    num_real = len(samples)
    assert num_real > 0, "need at least one sample"

    forward_bs = max(num_real, MIN_FORWARD_BS)
    padded = samples + [samples[0]] * (forward_bs - num_real)

    batch = _collate_samples(padded)
    batch = _move_to_device(batch, agent._device)
    batch["lang_goal"] = [[[s["lang_goal"]]] for s in padded]
    batch["tasks"] = [s["tasks"] for s in padded]

    action_gripper_pose = batch["gripper_pose"].float()  # (bs, 8)
    action_trans_con = action_gripper_pose[:, 0:3]
    action_rot = action_gripper_pose[:, 3:7]

    obs, pcd = _preprocess_inputs_memorybench(batch, list(cameras))
    pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)

    # Memory bundle (anchor + history) co-augmented with current pc so
    # stage 1's anchor / hist tokens and stage 2's anchor zoom render
    # land in the same frame as the current pc.
    mem_intermediate = None
    if (getattr(agent, "memory_enabled", False)
            and "anchor_mask" in padded[0]):
        K = int((getattr(agent._net_mod, "memory_cfg", {}) or {})
                .get("k_temporal", 2))
        mem_intermediate = agent._build_memory_inputs_from_replay(
            batch, list(cameras), K=K,
        )

    if apply_aug and getattr(agent, "_transform_augmentation", False):
        if mem_intermediate is not None:
            extras_for_aug = (
                [mem_intermediate["anchor_pc_batched"]]
                + list(mem_intermediate["hist_pc_batched"])
            )
            action_trans_con_np, action_rot_np, pc, perturbed_extras = (
                apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(agent.scene_bounds),
                    trans_aug_range=agent._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(agent._transform_augmentation_rpy),
                    extra_pcds=extras_for_aug,
                )
            )
            mem_intermediate["anchor_pc_batched"] = perturbed_extras[0]
            mem_intermediate["hist_pc_batched"] = perturbed_extras[1:]
        else:
            action_trans_con_np, action_rot_np, pc = apply_se3_aug_con(
                pcd=pc,
                action_gripper_pose=action_gripper_pose,
                bounds=torch.tensor(agent.scene_bounds),
                trans_aug_range=agent._transform_augmentation_xyz.clone().detach(),
                rot_aug_range=torch.tensor(agent._transform_augmentation_rpy),
            )
        action_trans_con = torch.tensor(action_trans_con_np).to(pc.device)
        action_rot = torch.tensor(action_rot_np).to(pc.device)

    pc, img_feat = rvt_utils.move_pc_in_bound(
        pc, img_feat, agent.scene_bounds, no_op=not agent.move_pc_in_bound,
    )

    memory_inputs = (
        agent._finalize_memory_inputs(mem_intermediate, pc)
        if mem_intermediate is not None else None
    )
    wpt = [x[:3] for x in action_trans_con]

    wpt_local_list = []
    for _pc, _wpt in zip(pc, wpt):
        a, _ = mvt_utils.place_pc_in_cube(
            _pc, _wpt,
            with_mean_or_bounds=agent._place_with_mean,
            scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
        )
        if getattr(agent, "align_real_frame", False):
            a = a.clone()
            a[..., 0:2] = -a[..., 0:2]
        wpt_local_list.append(a.unsqueeze(0))
    wpt_local = torch.cat(wpt_local_list, dim=0)

    pc = [mvt_utils.place_pc_in_cube(
        _pc,
        with_mean_or_bounds=agent._place_with_mean,
        scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
    )[0] for _pc in pc]
    if getattr(agent, "align_real_frame", False):
        pc = [_pc.clone() for _pc in pc]
        for _pc in pc:
            _pc[..., 0:2] = -_pc[..., 0:2]

    bs = len(pc)
    nc = agent._net_mod.num_img
    h = w = agent._net_mod.img_size

    # viz_force_gt_stage2=True: make mvt2 zoom into the GT waypoint (+noise)
    # the exact same way training does, instead of mvt1's own prediction.
    # This requires passing GT `wpt_local` here even though we're in eval mode.
    # Viz is still fully @torch.no_grad() + agent.eval(), so nothing trains.
    out = agent._net_mod(
        pc=pc,
        img_feat=img_feat,
        lang_emb=None,
        img_aug=0,
        wpt_local=wpt_local,
        rot_x_y=None,
        language_goal=batch["lang_goal"],
        viz_force_gt_stage2=True,
        memory_inputs=memory_inputs,
    )
    if memory_inputs is not None:
        out["_batch_anchor_mask"] = memory_inputs["anchor_mask"].detach()
    q_trans, _, _, _, _, _pts = agent.get_q(out, dims=(bs, nc, h, w))
    action_trans = agent.get_action_trans(
        wpt_local, _pts, out, None, dims=(bs, nc, h, w),
    )

    out["_batch_gripper_pose_orig"] = batch["gripper_pose"].float()
    aug_pose = torch.cat([action_trans_con, action_rot,
                          batch["gripper_pose"][:, 7:].float()], dim=-1)
    out["_batch_gripper_pose"] = aug_pose
    out["_batch_lang_goal"] = [s["lang_goal"] for s in padded]
    out["_batch_tasks"] = [s["tasks"] for s in padded]
    return out, q_trans, action_trans, bs, h, w


# memory-debug panel

@torch.no_grad()
def _render_memory_panel(agent, samples, cameras, apply_aug=True):
    """Render the per-step memory bundle the model attends to at stage 1:
    [anchor (frame 0), hist_0, ..., hist_{K-1}, current], for each sample
    in ``samples``. Same pipeline as the real training forward inside
    ``_update_shared`` — anchor / history are co-rotated with current under
    the same random SE3 so the panel matches what PaliGemma sees post-aug.
    Returns ``np.ndarray`` of shape ``(num_samples, K+2, num_views, H, W, 3)``
    uint8, or ``None`` if memory is disabled.
    """
    if not getattr(agent, "memory_enabled", False):
        return None
    if "anchor_mask" not in samples[0]:
        return None  # dataset wasn't built with memory_enabled=True

    forward_bs = max(len(samples), MIN_FORWARD_BS)
    padded = samples + [samples[0]] * (forward_bs - len(samples))
    batch = _collate_samples(padded)
    batch = _move_to_device(batch, agent._device)

    obs, pcd = _preprocess_inputs_memorybench(batch, list(cameras))
    pc_curr, img_feat_curr = rvt_utils.get_pc_img_feat(obs, pcd)

    K = int((getattr(agent._net_mod, "memory_cfg", {}) or {}).get("k_temporal", 2))
    mem_intermediate = agent._build_memory_inputs_from_replay(
        batch, list(cameras), K=K,
    )
    if mem_intermediate is None:
        return None

    if apply_aug and getattr(agent, "_transform_augmentation", False):
        action_gripper_pose = batch["gripper_pose"].float()
        extras_for_aug = (
            [mem_intermediate["anchor_pc_batched"]]
            + list(mem_intermediate["hist_pc_batched"])
        )
        _atc, _arq, pc_curr, perturbed_extras = apply_se3_aug_con(
            pcd=pc_curr,
            action_gripper_pose=action_gripper_pose,
            bounds=torch.tensor(agent.scene_bounds),
            trans_aug_range=agent._transform_augmentation_xyz.clone().detach(),
            rot_aug_range=torch.tensor(agent._transform_augmentation_rpy),
            extra_pcds=extras_for_aug,
        )
        mem_intermediate["anchor_pc_batched"] = perturbed_extras[0]
        mem_intermediate["hist_pc_batched"] = perturbed_extras[1:]

    pc_curr, img_feat_curr = rvt_utils.move_pc_in_bound(
        pc_curr, img_feat_curr, agent.scene_bounds,
        no_op=not agent.move_pc_in_bound,
    )
    memory_inputs = agent._finalize_memory_inputs(mem_intermediate, pc_curr)

    pc_curr_cube = [
        mvt_utils.place_pc_in_cube(
            _pc, with_mean_or_bounds=agent._place_with_mean,
            scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
        )[0]
        for _pc in pc_curr
    ]
    if getattr(agent, "align_real_frame", False):
        pc_curr_cube = [_pc.clone() for _pc in pc_curr_cube]
        for _pc in pc_curr_cube:
            _pc[..., 0:2] = -_pc[..., 0:2]

    mvt = agent._net_mod
    img_curr = mvt.render(
        pc=pc_curr_cube, img_feat=img_feat_curr,
        img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
    )
    img_anchor = mvt.render(
        pc=memory_inputs["anchor_pc"],
        img_feat=memory_inputs["anchor_img_feat"],
        img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
    )
    img_hists = [
        mvt.render(
            pc=memory_inputs["hist_pc"][k],
            img_feat=memory_inputs["hist_img_feat"][k],
            img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
        )
        for k in range(K)
    ]

    def _rgb(im):
        return im[:, :, 3:6, :, :].clamp(0, 1)

    anchor_mask = memory_inputs["anchor_mask"].view(-1, 1, 1, 1, 1)
    hist_mask = memory_inputs["hist_mask"]

    anchor_rgb = _rgb(img_anchor) * anchor_mask.to(torch.float32)
    hist_rgbs = [
        _rgb(img_hists[k]) * hist_mask[:, k].view(-1, 1, 1, 1, 1).to(torch.float32)
        for k in range(K)
    ]
    curr_rgb = _rgb(img_curr)

    rows = [anchor_rgb] + hist_rgbs + [curr_rgb]
    panel = torch.stack(rows, dim=1).permute(0, 1, 2, 4, 5, 3).contiguous()
    panel = (panel.cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return panel[: len(samples)]


def _save_memory_panel(panel_one_sample, save_path, view_names, step_label=""):
    """Save one sample's panel as a (K+2, V) grid PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, V, H, W, _ = panel_one_sample.shape
    K = rows - 2
    row_labels = (
        ["anchor (frame 0)"]
        + [f"hist k={k} (t-{k+1})" for k in range(K)]
        + ["current"]
    )

    fig, axes = plt.subplots(rows, V, figsize=(V * 3.0, rows * 3.0),
                             squeeze=False)
    for r in range(rows):
        for v in range(V):
            ax = axes[r][v]
            ax.imshow(panel_one_sample[r, v])
            ax.set_xticks([]); ax.set_yticks([])
            if v == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
            if r == 0:
                ax.set_title(view_names[v] if v < len(view_names) else f"view {v}",
                             fontsize=10)
    fig.suptitle(f"memory grid {step_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# public entry points

@torch.no_grad()
def visualize_samples(
    agent,
    samples: List[dict],
    save_dirs: List[str],
    stages: Sequence[str] = ("mvt1",),
    cameras: Sequence[str] = MEMORYBENCH_CAMERAS,
    extra_meta: Optional[List[dict]] = None,
    apply_aug: bool = False,
    forward_chunk: int = MIN_FORWARD_BS,
) -> None:
    assert len(samples) == len(save_dirs), "samples and save_dirs length mismatch"
    num_real = len(samples)
    if num_real == 0:
        return

    chunk = max(1, int(forward_chunk))

    was_training = agent._network.training
    agent.eval()
    try:
        for chunk_start in range(0, num_real, chunk):
            chunk_end = min(chunk_start + chunk, num_real)
            chunk_samples = samples[chunk_start:chunk_end]
            chunk_dirs = save_dirs[chunk_start:chunk_end]
            chunk_meta = (
                extra_meta[chunk_start:chunk_end]
                if extra_meta is not None else None
            )

            out, q_trans, action_trans, _bs, h, w = _run_viz_forward(
                agent, chunk_samples, cameras, apply_aug=apply_aug,
            )
            nc = agent._net_mod.num_img

            mem_panel = _render_memory_panel(
                agent, chunk_samples, cameras, apply_aug=apply_aug,
            )
            _renderer = getattr(agent._net_mod, "renderer", None)
            if getattr(_renderer, "oblique_views", False):
                _view_names = ["oblique_a", "oblique_b", "oblique_c"]
            else:
                _view_names = ["top", "front", "right"]

            view_logvar_all = None
            view_softmax_all = None
            if isinstance(out.get("view_logvar"), torch.Tensor):
                _lv = out["view_logvar"].detach().cpu().float()
                view_logvar_all = _lv.numpy()
                view_softmax_all = torch.softmax(-_lv, dim=-1).numpy()

            for i in range(len(chunk_samples)):
                sample_dir = chunk_dirs[i]
                os.makedirs(sample_dir, exist_ok=True)
                _use_focal = bool(getattr(agent, "use_modified_focal_loss", False))
                for stage in stages:
                    _save_stage(out, q_trans, action_trans, i, sample_dir, stage,
                                nc=nc, h=h, w=w,
                                use_modified_focal_loss=_use_focal)

                if mem_panel is not None:
                    _step_lbl = ""
                    if chunk_meta is not None and i < len(chunk_meta):
                        _step_lbl = (
                            f"(step {chunk_meta[i].get('step_idx', i)} / "
                            f"{chunk_meta[i].get('num_steps', '?')})"
                        )
                    _save_memory_panel(
                        mem_panel[i],
                        os.path.join(sample_dir, "memory_grid.png"),
                        view_names=_view_names,
                        step_label=_step_lbl,
                    )

                gp = out["_batch_gripper_pose"][i].cpu().tolist()
                gp_orig = out["_batch_gripper_pose_orig"][i].cpu().tolist()
                meta = {
                    "lang_goal":    out["_batch_lang_goal"][i],
                    "task":         out["_batch_tasks"][i],
                    "gt_xyz_m (aug)":     gp[0:3],
                    "gt_quat_xyzw (aug)": gp[3:7],
                    "gt_xyz_m (orig)":     gp_orig[0:3],
                    "gt_quat_xyzw (orig)": gp_orig[3:7],
                    "gt_claw":      gp[7],
                    "augmentation": getattr(agent, "_transform_augmentation", False),
                    "stages":       list(stages),
                }
                if view_logvar_all is not None:
                    meta["mvt1_view_s_i (raw log σ²)"] = [
                        round(float(x), 4) for x in view_logvar_all[i].tolist()
                    ]
                    meta["mvt1_view_softmax_p"] = [
                        round(float(x), 4) for x in view_softmax_all[i].tolist()
                    ]
                if chunk_meta is not None and i < len(chunk_meta):
                    meta.update(chunk_meta[i])
                _write_meta(sample_dir, meta)
    finally:
        if was_training:
            agent.train()


@torch.no_grad()
def visualize_epoch(
    agent,
    dataset,
    epoch: int,
    log_dir: str,
    cameras: Sequence[str] = MEMORYBENCH_CAMERAS,
    seed: Optional[int] = None,
    stages: Sequence[str] = ("mvt1",),
    forward_chunk: int = MIN_FORWARD_BS,
    tasks: Optional[Sequence[str]] = None,
) -> None:
    """Sample ONE episode per task and dump pred+GT viz for every step of it.

    Output layout::

        {log_dir}/viz/epoch_{epoch:04d}/
            {task}/episode{ep_idx:04d}/step_{k:03d}/{stage}/...

    Only rank 0 should call this — all forward ops go through
    ``agent._net_mod``; other ranks must not enter NCCL collectives between
    ``agent.eval()`` and the final ``agent.train()``.
    """
    assert len(dataset) > 0, "empty dataset"

    rng = np.random.default_rng(seed if seed is not None else epoch)
    # MemoryBench's dataset.index keys episodes by integer ``ep_idx``.
    groups = group_dataset_by_episode(
        dataset, task_field="task", episode_field="ep_idx",
    )
    if not groups:
        print(f"[viz] epoch {epoch}: empty dataset index, skipping")
        return

    if tasks is not None:
        whitelist = set(tasks)
        missing = whitelist - set(groups.keys())
        if missing:
            print(f"[viz] epoch {epoch}: viz_tasks contains unknown task(s) "
                  f"{sorted(missing)} — ignoring; available={sorted(groups.keys())}")
        groups = {t: eps for t, eps in groups.items() if t in whitelist}
        if not groups:
            print(f"[viz] epoch {epoch}: no tasks left after viz_tasks filter, skipping")
            return

    chosen = pick_one_episode_per_task(groups, rng)
    viz_root = os.path.join(log_dir, "viz", f"epoch_{epoch:04d}")

    # Tri-view layout for the stitched per-episode grids (rows=steps,
    # cols=views). Oblique renderer renames the same three slots.
    n_views = int(agent._net_mod.num_img)
    _renderer = getattr(agent._net_mod, "renderer", None)
    if getattr(_renderer, "oblique_views", False):
        view_names = ["oblique_a", "oblique_b", "oblique_c"]
    else:
        view_names = ["top", "front", "right"]

    total_steps = 0
    for task, ep_id, ds_indices in chosen:
        ep_name = f"episode{int(ep_id):04d}"
        num_steps = len(ds_indices)
        samples = [dataset[int(i)] for i in ds_indices]
        episode_dir = os.path.join(viz_root, task, ep_name)
        step_labels = [f"step_{k:03d}" for k in range(num_steps)]
        save_dirs = [os.path.join(episode_dir, s) for s in step_labels]
        extra_meta = [
            {
                "dataset_idx": int(ds_indices[k]),
                "task":        task,
                "ep_key":      ep_name,
                "ep_idx":      int(ep_id),
                "step_idx":    k,
                "num_steps":   num_steps,
            }
            for k in range(num_steps)
        ]

        visualize_samples(
            agent, samples, save_dirs,
            stages=stages, cameras=cameras, extra_meta=extra_meta,
            apply_aug=True, forward_chunk=forward_chunk,
        )

        # Stitch the per-step tri-view overlays into one big grid per stage:
        # rows = steps, cols = the 3 views.
        try:
            stitch_episode_overlays(
                episode_dir, step_labels, stages=stages,
                n_views=n_views, view_names=view_names,
            )
        except Exception as e:
            import traceback
            print(f"[viz] stitch failed task={task} ep={ep_name}: {e}")
            traceback.print_exc()

        total_steps += num_steps
        print(f"[viz] epoch {epoch}: task={task} ep={ep_name} "
              f"saved {num_steps} step(s)")

    print(f"[viz] epoch {epoch}: saved {total_steps} step(s) across "
          f"{len(chosen)} task(s) under {viz_root}")
