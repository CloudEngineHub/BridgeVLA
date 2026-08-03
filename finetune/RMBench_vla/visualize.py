"""
Dual-arm training-time visualization for RMBench.

For a sampled episode it runs the agent in eval mode over each keyframe and,
for BOTH arms, dumps the rendered tri-view + pred/GT heatmap overlays via the
shared ``_save_stage`` helper. Output layout:

    {log_dir}/viz/epoch_{e:04d}/{task}/{episode}/step_{k:03d}/arm_{L,R}/{mvt1,mvt2}/...

No point clouds are written to disk (only rendered views + heatmaps), per the
project's viz convention.

This mirrors memoryBench/visualize.py's forward plumbing but loops over arms
and consumes ``out["per_arm"]``. Kept intentionally small; safe to skip
(train.py wraps run_viz in try/except).
"""
import os
from typing import List, Optional

import numpy as np
import torch

from bridgevla.utils.viz_utils import (
    _save_memory_panel,
    _save_stage,
    group_dataset_by_episode,
    pick_one_episode_per_task,
    stitch_episode_overlays,
    write_rmbench_gripper_txt,
)
from bridgevla.mvt import utils as mvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con
import bridgevla.utils.rvt_utils as rvt_utils
import GemBench.utils.peract_utils_gembench as gembench_utils

MIN_FORWARD_BS = 4
ARM_NAMES = ["L", "R"]


def _to_tensor(a):
    return torch.from_numpy(np.asarray(a))


def _collate(samples):
    out = {}
    for k in samples[0].keys():
        v0 = samples[0][k]
        if isinstance(v0, dict):
            out[k] = {kk: torch.stack([_to_tensor(s[k][kk]) for s in samples], 0)
                      for kk in v0}
        elif isinstance(v0, str):
            out[k] = [s[k] for s in samples]
        else:
            out[k] = torch.stack([_to_tensor(s[k]) for s in samples], 0)
    return out


def _move(d, device):
    if isinstance(d, dict):
        return {k: _move(v, device) for k, v in d.items()}
    if isinstance(d, torch.Tensor):
        return d.to(device)
    return d


@torch.no_grad()
def _save_train_memory_grid(agent, memory_inputs, current_img_one,
                            save_path, view_names, step_label=""):
    """Render + save the memory keyframes actually FED to the model this step.

    Mirrors the eval-time ``memory_grid.png`` so train/eval are comparable.
    Rows = [anchor (frame 0)] + each VALID temporal-memory keyframe
    (slot 0 = most recent completed subtask) + [current]. Under the
    ``keyframe_gt`` policy these are exactly the GT subtask-boundary keyframes
    selected for ``step_idx`` by the dataset, after the SAME SE3 aug / cube
    projection the network sees — so this shows what PaliGemma's memory branch
    is given, not the raw dataset frames.

    ``memory_inputs`` is the post-``_finalize_memory_inputs`` dict (anchor_pc /
    hist_pc already cube-projected), ``current_img_one`` is the rendered
    current tri-view ``(V, 3, H, W)`` in [0, 1] (from ``out["mvt1_ori_img"]``).
    """
    if memory_inputs is None or not agent.memory_enabled:
        return
    hist_pc = memory_inputs.get("hist_pc")
    hist_feat = memory_inputs.get("hist_img_feat")
    if hist_pc is None:
        return
    anchor_pc = memory_inputs.get("anchor_pc")
    anchor_feat = memory_inputs.get("anchor_img_feat")
    anchor_mask = memory_inputs.get("anchor_mask")
    hist_mask = memory_inputs.get("hist_mask")     # (bs, K) bool
    K = len(hist_pc)

    def _render_one(pc_list, feat_list):
        # Render exactly as the model's memory branch does (stage-1, fixed
        # cameras, no aug). Use sample 0 (the viz batch repeats one sample).
        img = agent._net_mod.render(
            pc=pc_list, img_feat=feat_list, img_aug=0,
            mvt1_or_mvt2=True, dyn_cam_info=None,
        )
        return img[:, :, 3:6, :, :].clamp(0, 1)[0]   # (V, 3, H, W)

    rows = []
    row_labels = []

    # Anchor (frame 0). Blacked out when anchor_mask is False (step 0).
    a_valid = bool(anchor_mask[0].item()) if anchor_mask is not None else True
    anc_rgb = _render_one(anchor_pc, anchor_feat)
    if not a_valid:
        anc_rgb = torch.zeros_like(anc_rgb)
    rows.append(anc_rgb)
    row_labels.append("anchor (frame 0)")

    # Valid temporal-memory keyframes (slot 0 = most recent).
    n_valid = 0
    for k in range(K):
        valid = bool(hist_mask[0, k].item()) if hist_mask is not None else True
        if not valid:
            continue
        rows.append(_render_one(hist_pc[k], hist_feat[k]))
        row_labels.append(f"mem slot={k}")
        n_valid += 1

    # Current observation.
    rows.append(current_img_one.clamp(0, 1))
    row_labels.append("current")

    # (R, V, 3, H, W) -> (R, V, H, W, 3) uint8
    panel = torch.stack(rows, dim=0).permute(0, 1, 3, 4, 2).contiguous()
    panel = (panel.cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    _save_memory_panel(
        panel, save_path, view_names=view_names,
        step_label=step_label, row_labels=row_labels,
    )


@torch.no_grad()
def _viz_one_sample(agent, sample, cameras, save_dir, apply_se3_aug=True):
    """Run a dual-arm eval forward on a padded batch and dump per-arm stages.

    ``apply_se3_aug`` (default True): replicate the TRAINING-time SE3
    augmentation so the rendered scene shows the same random yaw rotation +
    translation the network is actually trained on. The SAME random SE3
    co-transforms (a) the shared scene point cloud, (b) BOTH arms' GT TCP
    positions (so the GT heatmap / stage-2 zoom stay aligned), and (c) the
    memory anchor (frame-0) AND every history-frame point cloud — so the
    fixed-camera per-pixel correspondence across memory frames is preserved,
    exactly as in ``_update_shared_dual``. Set False for the old canonical
    (un-augmented, epoch-comparable) view. Honors the training switch: aug is
    only applied when ``agent._transform_augmentation`` is also on. A fresh
    random SE3 is sampled per viz call, so the angle differs across epochs —
    that is intended (you are inspecting the augmentation, not a fixed pose).
    """
    device = agent._device
    samples = [sample] * MIN_FORWARD_BS
    batch = _move(_collate(samples), device)
    batch["lang_goal"] = [[[s]] for s in batch["lang_goal"]]

    obs, pcd = gembench_utils._preprocess_inputs_gembench(batch, cameras)
    pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)

    memory_inputs = None
    if agent.memory_enabled:
        K = int((getattr(agent._net_mod, "memory_cfg", {}) or {}).get("k_temporal", 4))
        memory_inputs = agent._build_memory_inputs_from_replay(batch, cameras, K=K)

    # ---- SE3 aug (mirror of _update_shared_dual). Same transform on the
    # shared cloud + arm0 (pivot) + arm1 + anchor/history memory PCs. ----
    arm_poses = [batch["left_action"], batch["right_action"]]
    arm_trans = [ap[:, 0:3] for ap in arm_poses]  # un-augmented default
    do_se3_aug = bool(apply_se3_aug) and bool(getattr(agent, "_transform_augmentation", False))
    if do_se3_aug:
        extras_for_aug = None
        if memory_inputs is not None:
            extras_for_aug = (
                [memory_inputs["anchor_pc_batched"]]
                + list(memory_inputs["hist_pc_batched"])
            )
        ret = apply_se3_aug_con(
            pcd=pc,
            action_gripper_pose=arm_poses[0],            # arm0 = primary pivot
            bounds=torch.tensor(agent.scene_bounds),
            trans_aug_range=agent._transform_augmentation_xyz.clone().detach(),
            rot_aug_range=torch.tensor(agent._transform_augmentation_rpy),
            extra_pcds=extras_for_aug,
            extra_action_poses=arm_poses[1:],
        )
        arm0_trans, _arm0_quat, pc = ret[0], ret[1], ret[2]
        idx = 3
        if extras_for_aug is not None:
            perturbed_extras = ret[idx]; idx += 1
            memory_inputs["anchor_pc_batched"] = perturbed_extras[0]
            memory_inputs["hist_pc_batched"] = perturbed_extras[1:]
        perturbed_extra_poses = ret[idx]
        arm_trans = [torch.tensor(arm0_trans).to(pc.device)]
        for (t_np, _q_np) in perturbed_extra_poses:
            arm_trans.append(torch.tensor(t_np).to(pc.device))

    pc, img_feat = rvt_utils.move_pc_in_bound(
        pc, img_feat, agent.scene_bounds, no_op=not agent.move_pc_in_bound)

    pc_cube, rev_trans = [], []
    for _pc in pc:
        a, b = mvt_utils.place_pc_in_cube(
            _pc, with_mean_or_bounds=False, scene_bounds=agent.scene_bounds)
        pc_cube.append(a)
        rev_trans.append(b)

    # Per-arm GT waypoint in the shared cube (for GT heatmap overlay) — uses
    # the (possibly SE3-perturbed) GT TCP position so it stays aligned.
    wpt_local_arms = []
    for at in arm_trans:
        wl = []
        for _pc, _wpt in zip(pc, at):
            ac, _ = mvt_utils.place_pc_in_cube(
                _pc, _wpt, with_mean_or_bounds=False, scene_bounds=agent.scene_bounds)
            wl.append(ac.unsqueeze(0))
        wpt_local_arms.append(torch.cat(wl, 0))

    memory_inputs = agent._finalize_memory_inputs(memory_inputs, pc)
    pc = pc_cube

    bs = len(pc)
    nc = agent._net_mod.num_img
    h = w = agent._net_mod.img_size

    # viz_force_gt_stage2=True + per-arm GT wpt_local: make mvt2 zoom into each
    # arm's GT crop (same as training) instead of mvt1's (early, wrong)
    # prediction. Without this, on an under-trained model mvt2 zooms to a bad
    # center and the GT TCP falls entirely outside the stage-2 crop — exactly
    # the "mvt2 GT not visible" symptom. Stage-2's in-crop heatmap is still the
    # model's own prediction (eval mode re-derives wpt from its heatmap), so
    # this only fixes the CROP, not the predicted peak. Mirrors
    # memoryBench/visualize.py's _run_viz_forward.
    out = agent._net_mod(
        pc=pc, img_feat=img_feat, img_aug=0,
        wpt_local=wpt_local_arms,  # per-arm GT -> stage-2 GT-centered zoom
        rot_x_y=[None, None],
        language_goal=batch["lang_goal"],
        viz_force_gt_stage2=True,
        memory_inputs=memory_inputs,
    )

    pred_grips = []
    for a_idx, out_arm in enumerate(out["per_arm"]):
        q_trans, _, grip_q, _, _, _ = agent.get_q(
            out_arm, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True,
        )
        pred_grips.append(int(grip_q[0].argmax().item()))
        action_trans = agent.get_action_trans(
            wpt_local_arms[a_idx], None, out_arm, None, dims=(bs, nc, h, w))
        arm_dir = os.path.join(save_dir, f"arm_{ARM_NAMES[a_idx]}")
        for stage in (("mvt1", "mvt2") if agent.stage_two else ("mvt1",)):
            _save_stage(
                out=out_arm, q_trans=q_trans, action_trans=action_trans,
                sample_idx=0, sample_dir=arm_dir, stage=stage,
                nc=nc, h=h, w=w,
                use_modified_focal_loss=agent.use_modified_focal_loss,
            )

    # ---- Memory grid: the keyframes fed to the model this step. ----
    if agent.memory_enabled and "mvt1_ori_img" in out:
        _renderer = getattr(agent._net_mod, "renderer", None)
        if getattr(_renderer, "oblique_views", False):
            _view_names = ["oblique_a", "oblique_b", "oblique_c"]
        else:
            _view_names = ["top", "front", "right"]
        try:
            _save_train_memory_grid(
                agent, memory_inputs,
                current_img_one=out["mvt1_ori_img"][0, :, 3:6],
                save_path=os.path.join(save_dir, "memory_grid.png"),
                view_names=_view_names,
            )
        except Exception as e:
            import traceback
            print(f"[RMBench viz] memory_grid failed: {e}")
            traceback.print_exc()

    left_gt = int(np.asarray(sample["left_action"])[-1])
    right_gt = int(np.asarray(sample["right_action"])[-1])
    write_rmbench_gripper_txt(
        save_dir,
        left_pred=pred_grips[0],
        right_pred=pred_grips[1] if len(pred_grips) > 1 else pred_grips[0],
        left_gt=left_gt,
        right_gt=right_gt,
    )


def visualize_epoch(agent, dataset, epoch, log_dir, cameras, seed=0,
                    stages=("mvt1", "mvt2"), tasks: Optional[List[str]] = None,
                    max_keyframes_per_episode: int = 0, apply_se3_aug: bool = True):
    """Pick one episode per task and dump dual-arm heatmaps for its keyframes.

    Default ``max_keyframes_per_episode=0`` => visualize EVERY keyframe of the
    sampled episode (full-trajectory rollout). Set it to a positive integer to
    cap how many keyframes are visualized per episode (evenly subsampled,
    always including the last) — useful if long-horizon RMBench tasks (e.g.
    blocks_ranking_try, ~100 keyframes) make the epoch-0 viz pass too slow,
    since each viz forward runs PaliGemma per arm per stage.

    ``apply_se3_aug`` (default True): render with the training-time SE3
    augmentation (random yaw + translation, applied identically to the scene
    cloud, both arms' GT, and all memory anchor/history clouds) so the viz
    reflects what the network actually sees during training. Set False for the
    old canonical un-augmented (cross-epoch comparable) view.
    """
    was_training = agent._network.training
    agent.eval()
    try:
        by_ep = group_dataset_by_episode(
            dataset, task_field="task", episode_field="ep_idx", step_field="step_idx",
        )
        rng = np.random.default_rng(seed)
        chosen = pick_one_episode_per_task(by_ep, rng=rng)

        # Tri-view layout for the stitched per-episode grids. The renderer
        # uses magic-angle oblique cameras when oblique_views=True; otherwise
        # the canonical top/front/right cube views.
        n_views = int(agent._net_mod.num_img)
        _renderer = getattr(agent._net_mod, "renderer", None)
        if getattr(_renderer, "oblique_views", False):
            view_names = ["oblique_a", "oblique_b", "oblique_c"]
        else:
            view_names = ["top", "front", "right"]
        arms = ARM_NAMES[:getattr(agent._net_mod, "num_arms", len(ARM_NAMES))]

        for (task, ep_idx, indices) in chosen:
            if tasks is not None and task not in tasks:
                continue
            # Subsample keyframes (evenly, keep the last) to bound viz cost.
            if max_keyframes_per_episode and max_keyframes_per_episode > 0 \
                    and len(indices) > max_keyframes_per_episode:
                step = max(1, len(indices) // max_keyframes_per_episode)
                sel = list(range(0, len(indices), step))[:max_keyframes_per_episode]
                if (len(indices) - 1) not in sel:
                    sel[-1] = len(indices) - 1
                indices = [indices[i] for i in sel]
            episode_dir = os.path.join(
                log_dir, "viz", f"epoch_{epoch:04d}", task, f"episode{ep_idx}")
            step_labels = []
            for step_k, idx in enumerate(indices):  # already step-ordered
                sample = dataset[idx]
                step_label = f"step_{step_k:03d}"
                save_dir = os.path.join(episode_dir, step_label)
                os.makedirs(save_dir, exist_ok=True)
                step_labels.append(step_label)
                try:
                    _viz_one_sample(agent, sample, cameras, save_dir,
                                    apply_se3_aug=apply_se3_aug)
                except Exception as e:
                    import traceback
                    print(f"[RMBench viz] failed {task} ep{ep_idx} step{step_k}: {e}")
                    traceback.print_exc()

            # Stitch the per-step tri-view overlays into one big grid per
            # (arm, stage): rows = steps, cols = the 3 views.
            try:
                stitch_episode_overlays(
                    episode_dir, step_labels, stages=stages,
                    n_views=n_views, view_names=view_names, arms=arms,
                )
            except Exception as e:
                import traceback
                print(f"[RMBench viz] stitch failed {task} ep{ep_idx}: {e}")
                traceback.print_exc()
    finally:
        if was_training:
            agent.train()
