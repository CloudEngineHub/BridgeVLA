"""Visualization helpers for the real-robot trainer & offline debug eval.

Two public entry points:

  - ``visualize_epoch(agent, dataset, ...)``   used by ``real/train.py`` at the
    start of every viz epoch (rank 0 only). Samples ONE episode per task from
    the training dataset and dumps pred+GT viz for every step in that episode
    in order. Replaces the earlier "pick 2 random transitions" behavior.

  - ``visualize_samples(agent, samples, save_dir, stages=("mvt1","mvt2"), ...)``
    general eval-side helper. Runs a single forward pass
    over the given samples in eval mode and dumps pred + GT heatmap overlays
    for the requested stages.

For every sample/stage we save, per 3-view ``i in {0,1,2}``:

    {sample_dir}/{stage}/original_{i}.png       (rendered RGB, saved once)
    {sample_dir}/{stage}/overlay_{i}.png        (pred | gt side-by-side)
    {sample_dir}/{stage}/logits_{i}.png         (pred | gt side-by-side)
    {sample_dir}/meta.txt
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
    _save_memory_panel,
    _save_mvt2_memory_grid,
    _save_stage,
    _write_meta,
    group_dataset_by_episode,
    pick_one_episode_per_task,
    stitch_episode_overlays,
)

from real.utils.peract_utils import _preprocess_inputs_real


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


def _collate_samples(samples: List[dict]) -> dict:
    """Stack a list of Real_Dataset samples into a batch dict.

    Can't use torch default_collate directly because the per-camera entries
    return numpy arrays; we just stack them tensor-by-tensor here.
    """
    assert len(samples) > 0
    keys = samples[0].keys()
    out: dict = {}
    for k in keys:
        v0 = samples[0][k]
        if isinstance(v0, dict):
            out[k] = {
                sub_k: torch.from_numpy(np.stack([s[k][sub_k] for s in samples], axis=0))
                for sub_k in v0
            }
        elif isinstance(v0, np.ndarray):
            out[k] = torch.from_numpy(np.stack([s[k] for s in samples], axis=0))
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
    """Run a single eval-mode forward through ``agent._net_mod`` (bypassing
    DDP) and return everything the visualization layer needs.

    When ``apply_aug=True`` the same SE3 augmentation used during training is
    applied so the rendered views match exactly what the model sees.

    Returns
    -------
    out          : dict produced by MVT.forward (contains mvt1_ori_img and,
                   when stage_two is on, mvt2_ori_img + the mvt2 sub-dict)
    q_trans      : (bs, h*w, nc*stage_count) translation logits
    action_trans : (bs, h*w, nc*stage_count) GT Gaussian heatmaps
    bs, h, w     : batch size (after padding) and image dims
    """
    num_real = len(samples)
    assert num_real > 0, "need at least one sample"

    # Pad to MIN_FORWARD_BS with copies of the first sample. Never saved.
    forward_bs = max(num_real, MIN_FORWARD_BS)
    padded = samples + [samples[0]] * (forward_bs - num_real)

    batch = _collate_samples(padded)
    batch = _move_to_device(batch, agent._device)
    batch["lang_goal"] = [[[s["lang_goal"]]] for s in padded]
    batch["tasks"] = [s["tasks"] for s in padded]

    # --- mirrors RVTAgent.update_gembench, with SE3 augmentation ---
    action_gripper_pose = batch["gripper_pose"]
    action_trans_con = action_gripper_pose[:, 0:3]
    action_rot = action_gripper_pose[:, 3:7]

    obs, pcd = _preprocess_inputs_real(batch, list(cameras))
    pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)

    # Episodic-memory bundle (anchor + history), co-augmented with the current
    # pc so stage-1 anchor/hist tokens and the stage-2 anchor zoom render land
    # in the same frame as the current pc. Mirrors GemBench/visualize.py.
    mem_intermediate = None
    if (getattr(agent, "memory_enabled", False)
            and "anchor_mask" in padded[0]):
        K = int((getattr(agent._net_mod, "memory_cfg", {}) or {})
                .get("k_temporal", 4))
        mem_intermediate = agent._build_memory_inputs_from_replay(
            batch, list(cameras), K=K,
        )

    # Apply the same SE3 augmentation used in training so the saved
    # images reflect exactly what the model sees during backprop. When memory
    # is on, the SAME transform is applied to anchor/history via extra_pcds.
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

    # Finalize memory_inputs while pc is still post-bound (the per-sample
    # reference for cube projection). _finalize_memory_inputs applies the same
    # cube projection + (agent.align_real_frame) flip the current frame gets.
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
    # This lets the viz show how far mvt2's heatmap peak drifts from GT under
    # the exact stage-2 crop mvt2 was trained on. Pass GT wpt_local even though
    # we're in eval mode — still @torch.no_grad() + agent.eval(), no training.
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
    q_trans, _, _, _, _, pts = agent.get_q(out, dims=(bs, nc, h, w))
    action_trans = agent.get_action_trans(
        wpt_local, pts, out, None, dims=(bs, nc, h, w),
    )
    # Carry the batch tensor back out for downstream meta.txt writing.
    # Store both original and (possibly augmented) gripper pose.
    out["_batch_gripper_pose_orig"] = batch["gripper_pose"]
    aug_pose = torch.cat([action_trans_con, action_rot,
                          batch["gripper_pose"][:, 7:]], dim=-1)
    out["_batch_gripper_pose"] = aug_pose
    out["_batch_lang_goal"] = [s["lang_goal"] for s in padded]
    out["_batch_tasks"] = [s["tasks"] for s in padded]
    return out, q_trans, action_trans, bs, h, w


# memory-debug panel

@torch.no_grad()
def _render_memory_panel(agent, samples, cameras, apply_aug=True):
    """Render the per-step memory bundle the model attends to at stage 1:
    [anchor (frame 0), hist_0, ..., hist_{K-1}, current], for each sample.
    Uses the SAME pipeline as the real training forward inside _update_shared:
    get_pc_img_feat -> apply_se3_aug_con(extra_pcds=...) -> move_pc_in_bound ->
    place_pc_in_cube(pc=current, app_pc=anchor) -> (align_real_frame flip) ->
    mvt1 renderer (fixed cameras). Returns (num_samples, K+2, V, H, W, 3) uint8
    with masked slots (step-0 anchor, missing history) blacked out, or None
    when memory is off / the dataset has no anchor bundle. Port of
    GemBench/visualize.py._render_memory_panel.
    """
    if not getattr(agent, "memory_enabled", False):
        return None
    if "anchor_mask" not in samples[0]:
        return None

    forward_bs = max(len(samples), MIN_FORWARD_BS)
    padded = samples + [samples[0]] * (forward_bs - len(samples))
    batch = _collate_samples(padded)
    batch = _move_to_device(batch, agent._device)

    obs, pcd = _preprocess_inputs_real(batch, list(cameras))
    pc_curr, img_feat_curr = rvt_utils.get_pc_img_feat(obs, pcd)

    K = int((getattr(agent._net_mod, "memory_cfg", {}) or {}).get("k_temporal", 4))
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

    def _rgb(im):  # (bs, V, channels, H, W) -> (bs, V, 3, H, W)
        return im[:, :, 3:6, :, :].clamp(0, 1)

    anchor_mask = memory_inputs["anchor_mask"].view(-1, 1, 1, 1, 1)
    hist_mask = memory_inputs["hist_mask"]  # (bs, K) bool

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


# public entry points

@torch.no_grad()
def visualize_samples(
    agent,
    samples: List[dict],
    save_dirs: List[str],
    stages: Sequence[str] = ("mvt1",),
    cameras: Sequence[str] = ("3rd",),
    extra_meta: Optional[List[dict]] = None,
    apply_aug: bool = False,
    forward_chunk: int = MIN_FORWARD_BS,
) -> None:
    """Run forward(s) over ``samples`` and dump pred+GT viz for each.

    ``samples`` and ``save_dirs`` must be the same length; one save_dir per
    real sample (padding copies are not written). ``stages`` subset of
    ``("mvt1", "mvt2")``. If ``stage_two`` is False on the agent, ``mvt2``
    requests are silently skipped.

    Long input lists are processed in chunks of ``forward_chunk`` samples to
    keep peak memory bounded during whole-episode visualization.
    """
    assert len(samples) == len(save_dirs), "samples and save_dirs length mismatch"
    num_real = len(samples)
    if num_real == 0:
        return

    chunk = max(1, int(forward_chunk))

    # Tri-view row labels for the memory grids (oblique renderer renames them).
    _renderer = getattr(agent._net_mod, "renderer", None)
    if getattr(_renderer, "oblique_views", False):
        _view_names = ["oblique_a", "oblique_b", "oblique_c"]
    else:
        _view_names = ["top", "front", "right"]

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

            # Memory-debug grid (anchor + K history + current), one per sample.
            # None when memory is disabled / the dataset has no anchor bundle.
            mem_panel = _render_memory_panel(
                agent, chunk_samples, cameras, apply_aug=apply_aug,
            )

            for i in range(len(chunk_samples)):
                sample_dir = chunk_dirs[i]
                os.makedirs(sample_dir, exist_ok=True)
                _use_focal = bool(getattr(agent, "use_modified_focal_loss", False))
                for stage in stages:
                    _save_stage(out, q_trans, action_trans, i, sample_dir, stage,
                                nc=nc, h=h, w=w,
                                use_modified_focal_loss=_use_focal)

                # Stage-1 memory grid: anchor + hist + current, all under
                # current's place_pc_in_cube transform so alignment is visible.
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

                # Stage-2 memory grid: anchor (frame 0) + current re-rendered
                # at the SAME zoomed-in stage-2 cameras, with the current row
                # overlaid by mvt2's predicted heatmap. Stage 2 has no temporal
                # block, only the spatial anchor — so two rows is the full
                # picture. Saved as {step}/mvt2/anchor_memory_grid.png so it
                # sits next to that step's mvt2 overlay_/logits_ panels,
                # matching GemBench/visualize.py's layout. Only emitted when
                # memory is active (out carries the anchor render) and stage 2
                # is being visualized.
                if (
                    "mvt2" in stages
                    and "mvt2_ori_img" in out
                    and "mvt2_anchor_ori_img" in out
                ):
                    _step_lbl = ""
                    if chunk_meta is not None and i < len(chunk_meta):
                        _step_lbl = (
                            f"(step {chunk_meta[i].get('step_idx', i)} / "
                            f"{chunk_meta[i].get('num_steps', '?')})"
                        )
                    _anchor_valid = True
                    if "_batch_anchor_mask" in out:
                        _anchor_valid = bool(out["_batch_anchor_mask"][i].item())
                    # mvt2 predicted heatmap for the current row. Post-
                    # activation matches the training objective (per-view
                    # sigmoid for the focal path, per-view spatial softmax
                    # otherwise), mirroring _save_stage so the overlay is
                    # directly comparable to overlay_{i}.png.
                    _mvt2_raw = q_trans[i, :, nc:2 * nc].clone().view(
                        h, w, nc
                    ).float()
                    if _use_focal:
                        _mvt2_hm = torch.sigmoid(_mvt2_raw).permute(
                            2, 0, 1
                        ).contiguous()
                    else:
                        _flat = _mvt2_raw.permute(2, 0, 1).reshape(nc, h * w)
                        _mvt2_hm = torch.softmax(_flat, dim=-1).view(nc, h, w)
                    _mvt2_dir = os.path.join(sample_dir, "mvt2")
                    os.makedirs(_mvt2_dir, exist_ok=True)
                    _save_mvt2_memory_grid(
                        anchor_img_one=out["mvt2_anchor_ori_img"][i, :, 3:6],
                        current_img_one=out["mvt2_ori_img"][i, :, 3:6],
                        current_hm_one=_mvt2_hm,
                        anchor_valid=_anchor_valid,
                        save_path=os.path.join(
                            _mvt2_dir, "anchor_memory_grid.png"
                        ),
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
    cameras: Sequence[str] = ("3rd",),
    seed: Optional[int] = None,
    stages: Sequence[str] = ("mvt1",),
    forward_chunk: int = MIN_FORWARD_BS,
    tasks: Optional[Sequence[str]] = None,
    # num_samples kept for backward-compat with older call sites; ignored now.
    num_samples: Optional[int] = None,
) -> None:
    """Sample ONE episode per task and dump pred+GT viz for every step of it.

    Output layout::

        {log_dir}/viz/epoch_{epoch:04d}/
            {task}/{episode_name}/step_{k:03d}/{stage}/...

    ``tasks`` whitelist (mirrors GemBench/visualize.py, driven by
    real_config.yaml's ``viz_tasks``):
        None  -> all tasks (default).
        list  -> only those tasks; unknown names are silently dropped.
        []    -> caller should already have skipped this call.

    Deterministic per ``seed`` (defaults to ``epoch``). All forward ops go
    through ``agent._net_mod`` so only rank 0 should call this — other ranks
    must not enter any NCCL collectives between ``agent.eval()`` and the
    final ``agent.train()``.
    """
    assert len(dataset) > 0, "empty dataset"
    del num_samples  # kept for call-site compat; sampling is now per-episode

    rng = np.random.default_rng(seed if seed is not None else epoch)
    groups = group_dataset_by_episode(
        dataset, task_field="task_name", episode_field="episode_name",
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
    for task, ep_name, ds_indices in chosen:
        num_steps = len(ds_indices)
        samples = [dataset[int(i)] for i in ds_indices]
        episode_dir = os.path.join(viz_root, task, ep_name)
        step_labels = [f"step_{k:03d}" for k in range(num_steps)]
        save_dirs = [os.path.join(episode_dir, s) for s in step_labels]
        extra_meta = [
            {
                "dataset_idx": int(ds_indices[k]),
                "task_name":   task,
                "episode_name": ep_name,
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
            print(f"[viz] stitch failed task={task} episode={ep_name}: {e}")
            traceback.print_exc()

        total_steps += num_steps
        print(f"[viz] epoch {epoch}: task={task} episode={ep_name} "
              f"saved {num_steps} step(s)")

    print(f"[viz] epoch {epoch}: saved {total_steps} step(s) across "
          f"{len(chosen)} task(s) under {viz_root}")
