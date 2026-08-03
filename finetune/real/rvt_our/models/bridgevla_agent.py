"""
BridgeVLA++ Agent for Real Robot Inference

Adapted from BridgeVLA/finetune/bridgevla/models/bridgevla_agent.py.
Training-only dependencies (yarr, RLBench, GemBench) are removed.
Only inference-relevant code is kept, plus a new `act_real()` method for
real robot deployment.

Refactored for real-robot eval.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel

# ── BridgeVLA model imports ─────────────────────────────────────────────────
# Ensure BridgeVLA/finetune is on sys.path *before* this module is loaded
# (done in the top-level eval scripts).
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.mvt.augmentation import aug_utils
# Item 5 ablation guard: shared with the training agent / MVT constructors so
# the deferred use_view_logvar branch fails fast here too (import-safe: pure
# torch-free helper, no yarr / RLBench / GemBench deps).
from bridgevla.mvt.view_logvar import assert_view_logvar_disabled
# Import-safe on the deployment machine: MemoryBank pulls in only torch +
# bridgevla.mvt.attn (no yarr / RLBench / GemBench), so it does not break the
# "no training deps" contract this fork exists to honor.
from bridgevla.mvt.memory import (
    MemoryBank,
    stack_hist_tokens_from_bank,
)


# Constants for real robot (override RLBench defaults)
# Fallback only — every current caller (rvt_our/eval_flask_app.py) passes
# scene_bounds=SCENE_BOUNDS_REAL from real/utils/peract_utils.py explicitly,
# so this is never actually hit. Kept in sync anyway so it isn't a landmine
# for a future caller that forgets to pass scene_bounds.
REAL_SCENE_BOUNDS = [
    -0.88, -0.46, -0.15,
    -0.02,  0.36,  0.55,
]
REAL_CAMERAS = ["3rd"]


def _align_real_frame_local(x):
    y = x.clone()
    y[..., 0:2] = -y[..., 0:2]
    return y


def _align_real_frame_rev_trans(rev_trans):
    def _rev_trans(x):
        return rev_trans(_align_real_frame_local(x))
    return _rev_trans


# Preprocessing helper (real robot observation → model input)
def _norm_rgb(x):
    """Normalise uint8 [0,255] → float [-1,1]."""
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_real(observation, cameras):
    """
    Convert a real-robot observation dict into the (obs, pcds) format
    expected by ``rvt_utils.get_pc_img_feat``.

    Args:
        observation: dict with camera keys, each containing 'rgb' and 'pcd'
                     tensors of shape (1, C, H, W).
        cameras: list of camera name strings, e.g. ["3rd"].

    Returns:
        obs:  list of [normed_rgb, pcd] per camera
        pcds: list of pcd tensors per camera
    """
    obs, pcds = [], []
    for cam in cameras:
        rgb = observation[cam]["rgb"]
        pcd = observation[cam]["pcd"]
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds


# Visualization helpers (used by act_real when return_views=True)
def _make_overlay(gray_vals: np.ndarray, orig_img: np.ndarray,
                  alpha: float = 0.5,
                  vmin: float = None, vmax: float = None) -> np.ndarray:
    """Blend a single-channel heatmap onto an RGB image. Returns uint8 (H,W,3).

    When vmin/vmax are provided the color mapping uses a fixed scale so that
    multiple images share the same color-to-value correspondence.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    hm = gray_vals.astype(np.float64)
    if vmin is None:
        vmin = hm.min()
    if vmax is None:
        vmax = hm.max()
    if vmax - vmin > 1e-8:
        hm = np.clip((hm - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        hm = np.zeros_like(hm)
    heatmap_rgb = (cm.jet(hm) * 255).astype(np.uint8)[..., :3]
    blended = (
        (1 - alpha) * orig_img.astype(np.float64)
        + alpha * heatmap_rgb.astype(np.float64)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


# RVTAgent – inference only
class RVTAgent:
    """BridgeVLA++ agent trimmed for real-robot inference.

    Constructor arguments mirror the training agent so that existing configs
    and ``load_agent`` still work.  Training-only paths are simply ignored.
    """

    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int = 72,
        stage_two: bool = False,
        move_pc_in_bound: bool = True,
        lr: float = 0.0001,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        weight_decay: float = 0.01,
        betas: list = [0.9, 0.95],
        warmup_steps: int = 2000,
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = None,
        cameras: list = None,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir: str = "",
        use_view_logvar: bool = False,
    ):
        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self._weight_decay = weight_decay
        self._betas = tuple(betas)
        self._warmup_steps = warmup_steps
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.stage_two = stage_two
        self.log_dir = log_dir
        self.scene_bounds = scene_bounds if scene_bounds is not None else REAL_SCENE_BOUNDS
        self.cameras = cameras if cameras is not None else REAL_CAMERAS
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug
        # Kendall-Gal per-view weighting (deferred ablation). Default False
        # (uniform 1/N averaging). True branch fails fast — mirrors the
        # training agent.
        assert_view_logvar_disabled(use_view_logvar)
        self.use_view_logvar = use_view_logvar

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        # Item 4 ablation: take the focal-loss flag from the MVT module.
        # Default False -> single up0 head + softmax+CE supervision.
        # Mirrors the shared training agent (bridgevla/models/bridgevla_agent.py).
        self.use_modified_focal_loss = bool(
            getattr(self._net_mod, "use_modified_focal_loss", False)
        )

        # Rotation/grip/collision feature source. When True (RMBench / real 6D
        # head), the MVT emits the rot/grip/coll ``feat`` from the STAGE-1
        # (coarse) heads, so get_q reads it from the top-level out dict instead
        # of out["mvt2"]. Mirrors the shared training agent + the MVT flag.
        self.feat_from_stage1 = bool(
            getattr(self._net_mod, "feat_from_stage1", False)
        )

        # rot_ver == 2 -> 6D continuous rotation regression head: the rotation
        # slice of the feat vector is a 6D vector (Zhou et al., CVPR 2019)
        # instead of 3*num_rotation_classes discrete logits, so num_all_rot is
        # 6 (feat layout [rot(6) | grip(2) | collision(2)]). Mirrors the shared
        # training agent so the same checkpoint decodes identically at deploy.
        self.rot_6d = (rot_ver == 2)
        self.num_all_rot = 6 if self.rot_6d else self._num_rotation_classes * 3

        # ---- Episodic memory (eval-time bank) ----
        # Mirrors the shared training agent (bridgevla/models/bridgevla_agent.py).
        # The bank accumulates the anchor (frame-0) + K most-recent keyframe
        # stage-1 PaliGemma tokens within an episode and is cleared by reset().
        # ``memory_enabled`` / ``k_temporal`` are read off the loaded MVT so the
        # bank's K always matches the network's built MemoryBlock K_temporal
        # (a mismatch would trip MemoryBlock.forward's assert). The whole block
        # is a no-op when the checkpoint's mvt_cfg has memory.enabled=False.
        self.memory_enabled = bool(getattr(self._net_mod, "memory_enabled", False))
        _net_mem_cfg = getattr(self._net_mod, "memory_cfg", {}) or {}
        self.memory_bank = MemoryBank(
            k_temporal=int(_net_mem_cfg.get("k_temporal", 4)),
            select=str(_net_mem_cfg.get("select", "sliding")),
        )
        # Cached anchor stage-1 tokens are extracted at step 0 under step 0's
        # place_pc_in_cube transform. That transform is only step-invariant
        # when place_with_mean=False (scene-bounds projection); with
        # place_with_mean=True every step uses its own mean/extent, so the
        # cached anchor would live in a different cube than later queries.
        # The real eval loader (eval_flask_app.load_agent) restores
        # place_with_mean=False for stage_two checkpoints, so this guard never
        # fires for a correctly-configured run — it's a tripwire against a
        # stage_two=False memory checkpoint silently producing misaligned KV.
        if self.memory_enabled and self._place_with_mean:
            raise ValueError(
                "memory.enabled=True is incompatible with place_with_mean=True: "
                "cached anchor tokens are extracted under step 0's mean/extent "
                "and would not align with later steps' cube. Use a stage_two "
                "checkpoint (place_with_mean restored to False at eval) or "
                "disable memory."
            )

    # Build / lifecycle
    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()

    def reset(self):
        # Episode boundary: clear the per-episode MemoryBank so anchor /
        # history tokens from a previous episode don't leak into the next.
        # The eval server calls this from its /reset endpoint, which the
        # client hits once at startup (a new client run == a new episode).
        # Harmless no-op when memory is disabled.
        self.memory_bank.reset()

    # Memory: project cached/anchor PCs into the current frame's cube
    def _project_memory_inputs_to_cube(self, memory_inputs, pc_post_bound):
        """Apply each sample's current-frame ``place_pc_in_cube`` transform to
        its anchor PC (via ``app_pc``), then the real-frame flip — so the
        anchor lands in the SAME cube and orientation as the current frame.

        Anchor-only: at eval, history is supplied as cached PaliGemma tokens
        (``hist_mvt1_tokens``), not raw PCs, so there is no ``hist_pc`` to
        project. The flip is applied UNCONDITIONALLY here to match the current
        frame's inline ``_align_real_frame_local`` (this fork does not use the
        shared agent's ``self.align_real_frame`` flag). Mirrors the shared
        agent's ``_project_memory_inputs_to_cube``.
        """
        if memory_inputs is None:
            return memory_inputs
        anchor_world = memory_inputs.get("anchor_pc")
        if anchor_world is not None:
            anchor_cube = []
            for i, _pc_curr in enumerate(pc_post_bound):
                a_cube, _ = mvt_utils.place_pc_in_cube(
                    _pc_curr, anchor_world[i],
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                anchor_cube.append(a_cube)
            # Unconditional flip — must match the current frame's flip exactly
            # (see _align_real_frame_local) so stage-2 anchor zoom-sync keeps
            # per-pixel correspondence.
            for _pc in anchor_cube:
                _pc[..., 0:2] = -_pc[..., 0:2]
            memory_inputs["anchor_pc"] = anchor_cube
        return memory_inputs

    # Memory-grid visualization snapshot
    def _build_memory_views(self, out, memory_inputs,
                            pc_world_for_bank, img_feat_for_bank):
        """Snapshot of the memory state that fed THIS step's forward.

        Mirrors the shared agent's ``act(visualize=True)`` memory-grid block:
        re-renders anchor + K history (slot 0 = most recent) + current through
        the SAME mvt1 pipeline the model uses (post-bound PC → cube → flip →
        render, fixed cameras, no aug). Invalid rows (empty anchor / hist
        slots) are blacked out. MUST be called BEFORE ``memory_bank.push()``
        so the current frame does not appear in its own history row.

        Returns a dict consumed by eval_flask_app for saving:
            panel        (K+2, V, H, W, 3) uint8 — rows [anchor, hist slot 0..K-1, current]
            row_labels   list[str] — slot 0 = most recent admitted keyframe
            anchor_valid bool                     — anchor_mask actually used
            n_hist       int                      — # valid history slots
            k_temporal   int                      — K
            mvt2_anchor  (V, 3, H, W) float cpu or None — stage-2 zoomed anchor
            mvt2_current (V, 3, H, W) float cpu or None — stage-2 zoomed current
            view_names   list[str]
        """
        K = self.memory_bank.k
        frames_pc, frames_feat, frame_valid, row_labels = [], [], [], []

        # Anchor (frame 0). At step 0 the slot is empty — render the current
        # frame as placeholder geometry but mark invalid (blacked out).
        if not self.memory_bank.first_frame():
            frames_pc.append(self.memory_bank.anchor_pc)
            frames_feat.append(self.memory_bank.anchor_img_feat)
            frame_valid.append(True)
        else:
            frames_pc.append(pc_world_for_bank)
            frames_feat.append(img_feat_for_bank)
            frame_valid.append(False)
        row_labels.append("anchor (frame 0)")

        # History slots: tensor slot k = k-th most recent. ``ordered_slots``
        # abstracts over both layouts (sliding: chronological reversed;
        # keyframe_gt: 2 recency frames + discriminator-admitted memory frames),
        # so this works whether select=sliding or keyframe_gt. Pad the remaining
        # slots up to K with blacked-out placeholders.
        slots = self.memory_bank.ordered_slots()
        n_hist = len(slots)
        for k in range(K):
            if k < n_hist and slots[k].get("pc") is not None:
                frames_pc.append(slots[k]["pc"])
                frames_feat.append(slots[k]["feat"])
                frame_valid.append(True)
            else:
                frames_pc.append(pc_world_for_bank)
                frames_feat.append(img_feat_for_bank)
                frame_valid.append(False)
            row_labels.append(
                f"mem slot={k}" if k < n_hist else f"hist k={k} (empty)"
            )

        # Current observation.
        frames_pc.append(pc_world_for_bank)
        frames_feat.append(img_feat_for_bank)
        frame_valid.append(True)
        row_labels.append("current")

        rendered_rgbs = []
        for f_pc, f_feat, valid in zip(frames_pc, frames_feat, frame_valid):
            pc_cube = []
            for _pc in f_pc:
                a, _ = mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                # Same real-frame flip as the current-frame forward path
                # (_align_real_frame_local clones, so bank PCs stay intact).
                pc_cube.append(_align_real_frame_local(a))
            img = self._net_mod.render(
                pc=pc_cube, img_feat=list(f_feat),
                img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
            )
            rgb = img[:, :, 3:6, :, :].clamp(0, 1)  # (bs, V, 3, H, W)
            if not valid:
                rgb = torch.zeros_like(rgb)
            rendered_rgbs.append(rgb[0])  # bs=1

        panel = torch.stack(rendered_rgbs, dim=0)        # (K+2, V, 3, H, W)
        panel = panel.permute(0, 1, 3, 4, 2).contiguous()
        panel = (panel.cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        anchor_valid = True
        if memory_inputs is not None and memory_inputs.get("anchor_mask") is not None:
            anchor_valid = bool(memory_inputs["anchor_mask"][0].item())

        # Stage-2 grid inputs: anchor re-rendered under the CURRENT step's
        # zoom vs the current zoomed render (per-pixel correspondence check).
        mvt2_anchor = None
        if "mvt2_anchor_ori_img" in out:
            mvt2_anchor = out["mvt2_anchor_ori_img"][0, :, 3:6].detach().cpu().float()
        mvt2_current = None
        if "mvt2_ori_img" in out:
            mvt2_current = out["mvt2_ori_img"][0, :, 3:6].detach().cpu().float()

        renderer = getattr(self._net_mod, "renderer", None)
        if getattr(renderer, "oblique_views", False):
            view_names = ["oblique_a", "oblique_b", "oblique_c"]
        else:
            view_names = ["top", "front", "right"]

        return {
            "panel": panel,
            "anchor_valid": anchor_valid,
            "n_hist": n_hist,
            "k_temporal": K,
            "mvt2_anchor": mvt2_anchor,
            "mvt2_current": mvt2_current,
            "view_names": view_names,
            "row_labels": row_labels,
        }

    # Q-value extraction
    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """Extract q-values from network output."""
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        # Rotation / grip / collision feature source:
        #   * feat_from_stage1=True (RMBench / real 6D head): the coarse stage-1
        #     heads emit ``feat`` (stage-1 global max-pool + stage-1 waypoint
        #     local patch), so read it from the top-level out dict; stage 2 then
        #     carries only the fine translation heatmap.
        #   * else: features live on stage 2 (forward_no_feat=False). When
        #     stage_two is False, fall back to top-level out.
        # Mirrors the shared training agent (bridgevla/models/bridgevla_agent.py).
        if self.feat_from_stage1:
            feat_out = out
        else:
            feat_out = out["mvt2"] if self.stage_two else out

        if get_q_trans:
            pts = None
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            if self.stage_two:
                q_trans2 = out["mvt2"]["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None

        if self.rot_ver in (0, 2):
            # rot_ver==0: rot_q is num_rotation_classes*3 discrete logits.
            # rot_ver==2: rot_q is the 6D rotation vector (num_all_rot==6).
            # Both share the contiguous [rot | grip(2) | collision(2)] layout.
            rot_q = feat_out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = feat_out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            collision_q = feat_out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
        elif self.rot_ver == 1:
            rot_q = torch.cat(
                (feat_out["feat_x"], feat_out["feat_y"], feat_out["feat_z"]), dim=-1
            ).view(bs, -1)
            grip_q = feat_out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = feat_out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            raise NotImplementedError(f"rot_ver={self.rot_ver}")

        y_q = None
        return q_trans, rot_q, grip_q, collision_q, y_q, pts

    # Prediction decoding
    def get_pred(self, out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info):
        """Decode network output into (wpt, rot_quat, grip, coll)."""
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        # NOTE: view_weights (per-view heatmap weighting) was reverted out of
        # the shared MVT.get_wpt ("come back to original" — uniform 1/N
        # averaging). Passing it here would TypeError, so we use the uniform
        # path, matching training.
        pred_wpt_local = self._net_mod.get_wpt(out, mvt1_or_mvt2, dyn_cam_info, y_q)

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        if self.rot_6d:
            # 6D regression: rot_q is (bs, 6) -> Gram-Schmidt -> R -> quat xyzw.
            # No argmax / discrete_euler_to_quaternion. Output matches the
            # discrete head's (bs, 4) numpy xyzw contract, so the downstream
            # deploy path (un-rotate + robot handoff) is unchanged.
            R_pred = aug_utils.rotation_6d_to_matrix(rot_q)  # (bs, 3, 3)
            pred_rot_quat = aug_utils.matrix_to_quaternion_xyzw_np(
                R_pred.detach().cpu().numpy()
            )
        else:
            pred_rot = torch.cat(
                (
                    rot_q[
                        :, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes
                    ].argmax(1, keepdim=True),
                    rot_q[
                        :, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes
                    ].argmax(1, keepdim=True),
                    rot_q[
                        :, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes
                    ].argmax(1, keepdim=True),
                ),
                dim=-1,
            )
            pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
                pred_rot.cpu(), self._rotation_resolution
            )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    # Real-robot inference
    @torch.no_grad()
    def act_real(self, observation: dict, cameras_view: list, return_views: bool = False):
        """
        Run a single inference step for a real-robot observation.

        Args:
            observation: dict with keys
                ``"language_goal"``  – nested list ``[[[text]]]``
                ``"<cam>"``          – dict with ``"rgb"`` and ``"pcd"``
                                       tensors of shape (1, 3, H, W)
            cameras_view: list of camera names, e.g. ``["3rd"]``
            return_views: if True, also return a dict of rendered view
                          images (original + overlay) for mvt1 and mvt2.

        Returns:
            target_pos     (np.ndarray, shape (3,))
            target_quat    (np.ndarray, shape (4,))
            target_gripper (np.ndarray, shape (1,))
            [views_info]   (dict) – only when return_views=True.
                keys: "mvt1" (and "mvt2" if stage_two), each mapping to
                {"originals": [img0,img1,img2], "overlays": [img0,img1,img2]}
                where each img is a uint8 (H, W, 3) numpy array.
        """
        language_goal = observation["language_goal"]

        obs, pcd = _preprocess_inputs_real(observation, cameras_view)
        pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # ---- Episodic memory: build memory_inputs from the bank state. ----
        # Mirrors the shared agent's act(). Stage-1 cameras are fixed across an
        # episode and the memory KV path is detached, so anchor / history
        # PaliGemma tokens are cached in the bank and reused verbatim every
        # step; only stage-2's anchor must be re-rendered each step under the
        # current zoom, for which we keep the anchor's raw post-bound PC.
        # Capture post-bound (world-ish) PCs BEFORE the per-sample cube
        # projection below clobbers ``pc``.
        pc_world_for_bank = [_pc.clone() for _pc in pc]
        img_feat_for_bank = [_f.clone() for _f in img_feat]

        memory_inputs = None
        if self.memory_enabled:
            _net_mem_cfg = getattr(self._net_mod, "memory_cfg", {}) or {}
            K = int(_net_mem_cfg.get("k_temporal", 4))
            bs_act = len(pc)
            first_frame = self.memory_bank.first_frame()

            if first_frame:
                # Episode start: anchor slot empty. Hand MVT the CURRENT PC as
                # an anchor placeholder + a zero anchor_mask — the spatial
                # MemoryBlock multiplies the anchor cross-attn residual by zero,
                # so step 0 is numerically identical to a no-memory step. The
                # current tokens are pushed below as the real anchor for later
                # steps.
                anchor_pc_world = [_p.clone() for _p in pc_world_for_bank]
                anchor_feat = [_f.clone() for _f in img_feat_for_bank]
                anchor_mask_t = torch.zeros(bs_act, dtype=torch.bool, device=self._device)
                anchor_tokens_t = None
            else:
                # Subsequent steps: reuse cached anchor tokens; raw PC kept for
                # stage-2 zoom re-rendering. Bank entries are already on device.
                anchor_pc_world = list(self.memory_bank.anchor_pc)
                anchor_feat = list(self.memory_bank.anchor_img_feat)
                anchor_mask_t = torch.ones(bs_act, dtype=torch.bool, device=self._device)
                anchor_tokens_t = self.memory_bank.anchor_mvt1_tokens

            # History: stack cached tokens into (bs, K, vlm_dim, V, H_p, W_p);
            # pad empty slots with zeros (mask gates them out).
            hist_tokens_t, hist_mask_t = stack_hist_tokens_from_bank(
                self.memory_bank, bs_act, K, self._device,
            )

            memory_inputs = {
                "anchor_pc": anchor_pc_world,
                "anchor_img_feat": anchor_feat,
                "anchor_mask": anchor_mask_t,
                "anchor_mvt1_tokens": anchor_tokens_t,
                # ``hist_pc`` / ``hist_img_feat`` intentionally absent at eval —
                # temporal memory is stage-1-only and uses cached tokens.
                "hist_mask": hist_mask_t,
                "hist_mvt1_tokens": hist_tokens_t,
            }
            # Project the anchor raw PC (used for stage-2 zoom) into the current
            # frame's cube + flip. History has no raw PC at eval.
            memory_inputs = self._project_memory_inputs_to_cube(memory_inputs, pc)

        # 2. place point cloud in unit cube
        pc_new, rev_trans = [], []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            a = _align_real_frame_local(a)
            b = _align_real_frame_rev_trans(b)
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        # 3. forward
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,
            language_goal=language_goal,
            memory_inputs=memory_inputs,
        )

        # 4. decode – also get q_trans when views are requested
        if return_views:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info,
        )

        # Un-rotate predicted quaternion from aligned frame back to real-robot frame
        _Rz180 = Rotation.from_euler('z', 180, degrees=True)
        for i in range(len(pred_rot_quat)):
            pred_rot_quat[i] = (_Rz180 * Rotation.from_quat(pred_rot_quat[i])).as_quat()

        result = (
            pred_wpt[0].cpu().numpy(),
            pred_rot_quat[0],
            pred_grip[0].cpu().numpy(),
        )

        # ---- Memory-grid snapshot (BEFORE the bank push) ----
        # Captures the bank state that actually fed this step's forward;
        # building it after push() would show the current frame inside its
        # own history row.
        memory_views = None
        if return_views and self.memory_enabled:
            memory_views = self._build_memory_views(
                out, memory_inputs, pc_world_for_bank, img_feat_for_bank,
            )

        # ---- Episodic memory: end-of-step bank update. ----
        # Push the current step's PRE-memory stage-1 PaliGemma tokens (detached)
        # into the bank, most-recent first. Raw post-bound PCs / img_feats are
        # kept only for an optional memory-grid visualization. Mirrors the
        # shared agent's act(). Runs before both return paths.
        if self.memory_enabled:
            curr_tokens = out.get("mvt1_paligemma_tokens", None)
            if curr_tokens is not None:
                curr_tokens = curr_tokens.detach()
            if self.memory_bank.first_frame():
                # Episode start: current tokens ARE the anchor tokens (the
                # spatial block was gated to identity by the zero anchor_mask
                # above, so no extra PaliGemma forward is needed at step 0).
                self.memory_bank.set_anchor(
                    [_p.detach() for _p in pc_world_for_bank],
                    [_f.detach() for _f in img_feat_for_bank],
                    curr_tokens,
                )

            # Keyframe-discriminator admission gate (mirrors the shared agent's
            # act()).
            #   * discriminator present -> admit iff sigmoid(logit) > threshold.
            #     For the real press-button tasks the head is trained (BCE) to
            #     fire on the completed-press boundary frames, so the gate
            #     admits those into the extra memory slots; for tasks with no
            #     boundary label it stays effectively False.
            #   * keyframe_gt WITHOUT a discriminator -> NEVER admit, so the
            #     temporal memory stays the frame-0 anchor + the two most-recent
            #     executed keyframes (recency slots).
            #   * sliding (legacy) -> admit all (gate=True), unchanged.
            gate = True
            mem_logit = out.get("mem_logit", None)
            if mem_logit is not None:
                _disc_cfg = (getattr(self._net_mod, "memory_cfg", {}) or {}).get(
                    "discriminator", {}) or {}
                _thr = float(_disc_cfg.get("threshold", 0.5))
                gate = bool(torch.sigmoid(mem_logit).mean().item() > _thr)
            elif getattr(self.memory_bank, "select", "sliding") == "keyframe_gt":
                gate = False

            self.memory_bank.push(
                curr_tokens,
                gate=gate,
                pc_world=[_p.detach() for _p in pc_world_for_bank],
                img_feat=[_f.detach() for _f in img_feat_for_bank],
            )

        if return_views:
            views_info = {}
            stages = [("mvt1", "mvt1_ori_img", slice(0, nc))]
            if self.stage_two:
                stages.append(("mvt2", "mvt2_ori_img", slice(nc, 2 * nc)))

            for stage_name, img_key, ch_slice in stages:
                if img_key not in out:
                    continue
                rgb = out[img_key][0, :, 3:6].float()  # (nc, 3, H, W) in [0,1]
                color_imgs = rgb.cpu().numpy().transpose(0, 2, 3, 1)  # (nc, H, W, 3)

                pred_raw = q_trans[0, :, ch_slice].clone().view(h, w, nc).float()
                # Heatmap activation — mirrors the training viz
                # (_save_stage / act(visualize=True)). When
                # use_modified_focal_loss=False (default, CE loss) the
                # training objective is a per-view spatial softmax over (H*W);
                # sigmoid is only correct for the CenterNet focal-loss path.
                if self.use_modified_focal_loss:
                    # CenterNet path: per-pixel sigmoid → independent Bernoulli probs.
                    pred_hm = torch.sigmoid(pred_raw)
                else:
                    # Original BridgeVLA: spatial softmax over (H*W) per view.
                    _flat = pred_raw.permute(2, 0, 1).reshape(nc, h * w)
                    _sm = torch.softmax(_flat, dim=-1)
                    pred_hm = _sm.view(nc, h, w).permute(1, 2, 0).contiguous()
                pred_gray = pred_hm.cpu().numpy().transpose(2, 0, 1)  # (nc, H, W)
                pred_logits = pred_raw.cpu().numpy().transpose(2, 0, 1)  # (nc, H, W)

                originals, overlays, logits = [], [], []
                for i in range(nc):
                    orig = (np.clip(color_imgs[i], 0, 1) * 255).astype(np.uint8)
                    originals.append(orig)
                    # Per-view min/max (no vmin/vmax), matching the training
                    # viz (viz_utils._save_stage) and the SAM2Act eval server.
                    # A scale shared across the three views used to be used
                    # here, but it renders a diffuse view uniformly dark; the
                    # "which view is more confident" signal lives in the
                    # logits_{i}.png panels, which DO share a scale.
                    overlays.append(_make_overlay(pred_gray[i], orig))
                    logits.append(pred_logits[i].astype(np.float64))

                views_info[stage_name] = {
                    "originals": originals,
                    "overlays": overlays,
                    "logits": logits,
                }

            if memory_views is not None:
                views_info["memory"] = memory_views

            return result + (views_info,)

        return result

    # Pretty printing
    def __repr__(self):
        return (
            f"RVTAgent(\n"
            f"  rot_ver={self.rot_ver}, stage_two={self.stage_two},\n"
            f"  num_rotation_classes={self._num_rotation_classes},\n"
            f"  scene_bounds={self.scene_bounds},\n"
            f"  cameras={self.cameras},\n"
            f"  place_with_mean={self._place_with_mean},\n"
            f"  move_pc_in_bound={self.move_pc_in_bound},\n"
            f")"
        )
