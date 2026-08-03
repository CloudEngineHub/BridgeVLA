'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/models/rvt_agent.py
Therefore, the code is also under the NVIDIA Source Code License

'''

import pprint
import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
import RLBench.utils.peract_utils_rlbench as rlbench_utils
import GemBench.utils.peract_utils_gembench as gembench_utils
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con, aug_utils
from bridgevla.mvt.heads_focal import (
    HM_MIN_RADIUS, HM_VLA_MIN_RADIUS,
    generate_centernet_hm_from_bbox, generate_centernet_hm_from_pt,
    modified_focal_loss, modified_focal_loss_per_heatmap,
)
from bridgevla.mvt.view_logvar import assert_view_logvar_disabled
from bridgevla.mvt.memory import MemoryBank, stack_hist_tokens_from_bank
from bridgevla.utils.viz_utils import (
    _save_anchor_frames,
    _save_memory_panel,
    _save_mvt2_memory_grid,
    _save_stage,
    stitch_episode_overlays,
    write_rmbench_gripper_txt,
    write_rmbench_zoom_pt_count_txt,
    write_rmbench_memory_trigger_txt,
)
from yarr.agents.agent import ActResult
from PIL import Image, ImageDraw


def _rmbench_stage2_force_center() -> bool:
    """Eval-only: skip mvt2 translation argmax; use zoom center (wpt_local1)."""
    return os.environ.get("stage2_force_center", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _rmbench_stage2_sparsepc_force_center() -> bool:
    """Eval-only: when zoom is sparse, force translation to zoom center (stage1 waypoint)."""
    return os.environ.get("stage2_sparsePC_force_center", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
import torch
import numpy as np
import os


def save_point_cloud_with_color(filename, points, colors, keypoint=None):
    """
    Save the point cloud and colors to a PLY file, automatically handling the color value range.
    :param filename: Output file name (e.g. 'point_cloud.ply')
    :param points: Point cloud coordinates (N,3) np.array
    :param colors: Color values (N,3) np.array (0-255 or 0-1)
    :param keypoint: Keypoint coordinates (3,) np.array (optional)
    """

    assert points.shape[1] == 3 
    assert colors.shape[1] == 3
    
    if colors.max() <= 1.0:  # If color values are between 0-1
        colors = (colors * 255).astype(np.uint8)
    else:  # If color values are between 0-255
        colors = colors.astype(np.uint8)
    
    if keypoint is not None:
        points = np.vstack([points, keypoint])
        colors = np.vstack([colors, np.array([255, 0, 0])])  # Mark keypoint in red

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, clr in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def _loss_scalar(x):
    return float(x.item() if torch.is_tensor(x) else x)


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


class RVTAgent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        move_pc_in_bound: bool,
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
        predict_collision: bool = True,
        scene_bounds: list = rlbench_utils.SCENE_BOUNDS,
        cameras: list = rlbench_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
        align_real_frame: bool = False,
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
        # ``predict_collision`` gates the per-step ignore-collisions SUPERVISION, not the head width: the
        # feat head always emits grip(2)+collision(2) so checkpoints stay loadable across benches.
        # False (GemBench / memoryBench / RMBench) = no usable label, CE dropped, logits untrained. Eval on
        # those benches ignores the slot regardless, so an old True-trained ckpt still evaluates identically.
        self.predict_collision = bool(predict_collision)
        self.stage_two = stage_two
        self.log_dir = log_dir
        self.scene_bounds = scene_bounds
        self.cameras = cameras

        print("Cameras:",self.cameras)
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug
        self.align_real_frame = align_real_frame
        # Kendall-Gal per-view weighting (deferred ablation); the True branch raises NotImplementedError.
        assert_view_logvar_disabled(use_view_logvar)
        self.use_view_logvar = use_view_logvar

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        # Focal-loss flag taken from the MVT module. Default False -> single up0 head + softmax+CE.
        self.use_modified_focal_loss = bool(
            getattr(self._net_mod, "use_modified_focal_loss", False)
        )

        # feat_from_stage1 (RMBench): rot/grip/coll ``feat`` comes from the STAGE-1 heads, so get_q reads it
        # from the top-level out dict instead of out["mvt2"]. Mirrors the MVT module's own flag.
        self.feat_from_stage1 = bool(
            getattr(self._net_mod, "feat_from_stage1", False)
        )

        # rot_ver == 2 -> 6D continuous rotation regression head (RMBench): the feat vector's rotation slice
        # is a 6D vector (Zhou et al., CVPR 2019) rather than 3 * num_rotation_classes logits, so num_all_rot is 6.
        self.rot_6d = (rot_ver == 2)
        self.num_all_rot = 6 if self.rot_6d else self._num_rotation_classes * 3

        # ---- Episodic memory ----
        # The bank is the eval-time accumulator; training pulls anchor / history straight from the dataset.
        # Built unconditionally and a no-op when memory_cfg.enabled=False on the network.
        self.memory_enabled = bool(getattr(self._net_mod, "memory_enabled", False))
        _net_mem_cfg = getattr(self._net_mod, "memory_cfg", {}) or {}
        self.memory_bank = MemoryBank(
            k_temporal=int(_net_mem_cfg.get("k_temporal", 2)),
            select=str(_net_mem_cfg.get("select", "sliding")),
        )
        # Set in act() so the next step's slot 0 carries the action predicted at the previous step.
        self._last_predicted_action = None

        # ---- Eval overlay-viz episode tracking ----
        # act(visualize=True) records the per-step viz dir + step labels so finalize_eval_viz() can stitch the
        # overlay tri-views into per-stage grids. Single-arm counterpart of RMBench's act_rmbench bookkeeping.
        self._eval_viz_episode_dir = None
        self._eval_viz_step_labels = []
        self._eval_viz_instruction = None

        # ---- Memory consistency guards ----
        # The eval cache reuses anchor stage-1 tokens from step 0, so ``place_pc_in_cube`` must be identical
        # every step; place_with_mean=True varies per frame and is rejected at construction.
        if self.memory_enabled and self._place_with_mean:
            raise ValueError(
                "memory.enabled=True is incompatible with rvt.place_with_mean=True: "
                "cached anchor tokens are extracted at step 0 under step 0's mean/extent "
                "and would not align with step t's per-frame cube. "
                "Set rvt.place_with_mean: false in the GemBench yaml."
            )
        # img_aug / img_aug_2 draw fresh noise on every render. Anchor / history render with img_aug=0, so a
        # non-zero value breaks per-pixel correspondence with the memory frames. Warn loudly (GemBench default is 0).
        if self.memory_enabled:
            _img_aug_main = float(self.img_aug) if self.img_aug else 0.0
            _img_aug_2 = float(getattr(self._net_mod, "img_aug_2", 0.0))
            if _img_aug_main != 0.0 or _img_aug_2 != 0.0:
                import warnings
                warnings.warn(
                    f"memory.enabled=True with img_aug={_img_aug_main} / "
                    f"img_aug_2={_img_aug_2}: the noise is applied only to the "
                    f"current frame (anchor / history use img_aug=0), so per-pixel "
                    f"correspondence between current and memory KV is broken. "
                    f"Set rvt.img_aug=0 and mvt_cfg.img_aug_2=0 to match."
                )

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
        self._build_optimizer()
        self._global_step = 0

    def _build_optimizer(self):
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self._network.parameters())
        )
        if self._optimizer_type == "adamw":
            self._optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self._lr,
                weight_decay=self._weight_decay,
                betas=self._betas,
            )
        else:
            self._optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )

    def rebuild_optimizer(self):
        """Rebuild optimizer after freezing/unfreezing parameters (stage switch)."""
        self._build_optimizer()
        self._global_step = 0

    def _get_warmup_lr_scale(self):
        """Linear warmup: scale from 0 to 1 over warmup_steps."""
        if self._warmup_steps <= 0:
            return 1.0
        return min(1.0, self._global_step / self._warmup_steps)

    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs, 1), or None
            when the bench has no collision label (``predict_collision=False``).
            None leaves ``action_collision_one_hot`` all-zero; it is never
            consumed in that case (the CE term is skipped in `_update_shared`).
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        for b in range(bs):
            # 6D regression head (rot_ver==2): rotation is not discretized, the rot one-hots stay zero, and
            # the 6D loss reads the GT quaternion -> rotation matrix directly via _gt_rotmat_from_quat.
            if not self.rot_6d:
                gt_rot = action_rot[b]
                gt_rot = aug_utils.quaternion_to_discrete_euler(
                    gt_rot, self._rotation_resolution
                )
                action_rot_x_one_hot[b, gt_rot[0]] = 1
                action_rot_y_one_hot[b, gt_rot[1]] = 1
                action_rot_z_one_hot[b, gt_rot[2]] = 1

            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision (skipped when the bench supplies no label; the all-zero one-hot is unused).
            if action_ignore_collisions is not None:
                gt_ignore_collisions = action_ignore_collisions[b, :]
                action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def _gt_rotmat_from_quat(self, action_rot, device):
        """(bs, 4) quaternion xyzw (np.ndarray or torch.Tensor) -> (bs, 3, 3)
        torch rotation matrix, used as the GT target for the 6D regression
        head (rot_ver==2).

        No discretization / gimble_fix / hemisphere handling: the matrix is
        the exact GT rotation (R and the antipodal quaternion -q yield the
        same R, so the 6D target is sign-invariant by construction).
        """
        if isinstance(action_rot, torch.Tensor):
            action_rot = action_rot.detach().cpu().numpy()
        R = aug_utils.quaternion_xyzw_to_matrix_np(action_rot)  # (bs, 3, 3)
        return torch.from_numpy(np.asarray(R)).float().to(device)

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction

        Heatmaps come from both stages (stage1 ++ stage2) when stage_two.
        Rotation / gripper / collision features come from stage 2 — stage 1
        runs ``forward_no_feat=True`` and only emits ``trans``. This matches
        the original BridgeVLA architecture.
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans from stage1 and stage2
            if self.stage_two:
                q_trans2 = out["mvt2"]["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None

        # Rotation / grip / collision feature source: with feat_from_stage1=True (RMBench) the coarse stage-1
        # heads emit ``feat`` at the top level and stage 2 carries only the fine translation heatmap.
        if self.feat_from_stage1:
            feat_out = out
        else:
            feat_out = out["mvt2"] if self.stage_two else out

        if self.rot_ver in (0, 2):
            # rot_q is discrete logits (rot_ver==0) or the 6D vector (rot_ver==2); layout is [rot | grip(2) | collision(2)].
            rot_q = feat_out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = feat_out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            collision_q = feat_out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
        elif self.rot_ver == 1:
            rot_q = torch.cat((feat_out["feat_x"], feat_out["feat_y"], feat_out["feat_z"]),
                              dim=-1).view(bs, -1)
            grip_q = feat_out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = feat_out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts


    def _build_memory_inputs_from_replay(self, replay_sample, cameras, K):
        """Build a memory_inputs INTERMEDIATE dict from a GemBench replay
        sample. Returns the anchor / history point clouds in **batched**
        form (post ``get_pc_img_feat``, PRE ``move_pc_in_bound``), plus
        per-slot validity masks. Historical actions are NOT consumed
        (``MemoryBlock.action_proj`` was removed; temporal memory is
        purely visual + per-slot index PE).

        The batched form is required so SE3 augmentation in
        ``_update_shared`` can perturb current AND memory PCs with the
        SAME transform (the rotation pivot = current step's gripper pose,
        the translation shift is shared). After SE3, the extras are
        passed through ``_finalize_memory_inputs`` to do
        ``move_pc_in_bound`` -> cube projection (via ``app_pc`` from
        current) -> optional align_real_frame, producing the final dict
        consumed by ``MVT.forward``.

        Returns None when the dataset didn't supply memory bundles.
        """
        if "anchor_mask" not in replay_sample:
            return None

        device = self._device

        # Anchor batched.
        anchor_obs, anchor_pcd = gembench_utils._preprocess_inputs_gembench_prefixed(
            replay_sample, cameras, prefix="anchor_",
        )
        anchor_pc_b, anchor_feat_b = rvt_utils.get_pc_img_feat(
            anchor_obs, anchor_pcd,
        )

        # History batched (K slots).
        hist_pc_b = []
        hist_feat_b = []
        for k in range(K):
            obs_k, pcd_k = gembench_utils._preprocess_inputs_gembench_prefixed(
                replay_sample, cameras, prefix=f"hist{k}_",
            )
            pc_k, feat_k = rvt_utils.get_pc_img_feat(obs_k, pcd_k)
            hist_pc_b.append(pc_k)
            hist_feat_b.append(feat_k)

        anchor_mask = replay_sample["anchor_mask"]
        if not isinstance(anchor_mask, torch.Tensor):
            anchor_mask = torch.tensor(anchor_mask, device=device)
        anchor_mask = anchor_mask.to(device).bool()

        hist_mask = replay_sample["hist_mask"]
        if not isinstance(hist_mask, torch.Tensor):
            hist_mask = torch.tensor(hist_mask, device=device)
        hist_mask = hist_mask.to(device).bool()

        return {
            # Batched form (bs, N, 3) / (bs, N, F): perturb_se3-friendly.
            "anchor_pc_batched": anchor_pc_b,
            "anchor_feat_batched": anchor_feat_b,
            "hist_pc_batched": hist_pc_b,           # list of K Tensors
            "hist_feat_batched": hist_feat_b,       # list of K Tensors
            # Meta (already on device).
            "anchor_mask": anchor_mask,
            "hist_mask": hist_mask,
        }

    def _finalize_memory_inputs(self, mem_intermediate, current_pc_post_bound):
        """Convert the batched intermediate dict (after any SE3 aug has
        been applied to its PCs in ``_update_shared``) into the LIST-form
        dict that ``MVT.forward`` consumes:

            anchor_pc / anchor_img_feat: list of per-sample tensors,
                projected into the SAME cube as the current frame via
                ``place_pc_in_cube(pc=current, app_pc=anchor)``;
            hist_pc / hist_img_feat:   list of K such per-sample lists;
            anchor_mask, hist_mask: meta tensors.

        Mirrors the cube-projection + align_real_frame logic the current
        frame's PCs go through.
        """
        if mem_intermediate is None:
            return None

        # Batched -> list (move_pc_in_bound culls per-sample).
        anchor_pc, anchor_img_feat = rvt_utils.move_pc_in_bound(
            mem_intermediate["anchor_pc_batched"],
            mem_intermediate["anchor_feat_batched"],
            self.scene_bounds, no_op=not self.move_pc_in_bound,
        )
        K = len(mem_intermediate["hist_pc_batched"])
        hist_pc = []
        hist_img_feat = []
        for k in range(K):
            h_pc, h_ft = rvt_utils.move_pc_in_bound(
                mem_intermediate["hist_pc_batched"][k],
                mem_intermediate["hist_feat_batched"][k],
                self.scene_bounds, no_op=not self.move_pc_in_bound,
            )
            hist_pc.append(h_pc)
            hist_img_feat.append(h_ft)

        memory_inputs = {
            "anchor_pc": anchor_pc,
            "anchor_img_feat": anchor_img_feat,
            "anchor_mask": mem_intermediate["anchor_mask"],
            "hist_pc": hist_pc,
            "hist_img_feat": hist_img_feat,
            "hist_mask": mem_intermediate["hist_mask"],
        }
        # Apply current's per-frame ``place_pc_in_cube`` to anchor / hist via ``app_pc`` so all share one cube.
        memory_inputs = self._project_memory_inputs_to_cube(
            memory_inputs, current_pc_post_bound,
        )
        return memory_inputs

    def _project_memory_inputs_to_cube(self, memory_inputs, pc_post_bound):
        """Apply each batch sample's current-frame ``place_pc_in_cube``
        transform to its anchor and history PCs (via ``app_pc``).
        Operates in-place on ``memory_inputs``. Must be called BEFORE the
        current ``pc`` is overwritten by its cube projection so we still
        have the per-sample reference PC.

        At eval, history is supplied as cached PaliGemma tokens (not raw
        PCs) — ``hist_pc`` is absent and the history projection is skipped.
        Training keeps the raw-PC path for both anchor and history.
        """
        if memory_inputs is None:
            return memory_inputs
        bs = len(pc_post_bound)
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
            memory_inputs["anchor_pc"] = anchor_cube

        hist_world = memory_inputs.get("hist_pc")
        if hist_world is not None:
            K = len(hist_world)
            hist_cube = [[] for _ in range(K)]
            for k in range(K):
                for i, _pc_curr in enumerate(pc_post_bound):
                    h_cube, _ = mvt_utils.place_pc_in_cube(
                        _pc_curr, hist_world[k][i],
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )
                    hist_cube[k].append(h_cube)
            memory_inputs["hist_pc"] = hist_cube

        if self.align_real_frame:
            if memory_inputs.get("anchor_pc") is not None:
                for _pc in memory_inputs["anchor_pc"]:
                    _pc[..., 0:2] = -_pc[..., 0:2]
            if memory_inputs.get("hist_pc") is not None:
                for slot_pcs in memory_inputs["hist_pc"]:
                    for _pc in slot_pcs:
                        _pc[..., 0:2] = -_pc[..., 0:2]

        return memory_inputs

    def _save_eval_memory_grid(
        self,
        save_dir,
        step_label,
        pc_world_for_bank,
        img_feat_for_bank,
    ):
        """Re-render anchor + temporal memory + current for eval ``memory_grid.png``.

        Must run BEFORE ``memory_bank.push()`` so the current frame does not
        appear in its own history row. Tensor slot 0 = most recent keyframe.
        """
        if not self.memory_enabled:
            return
        os.makedirs(save_dir, exist_ok=True)
        _renderer = getattr(self._net_mod, "renderer", None)
        if getattr(_renderer, "oblique_views", False):
            view_names = ["oblique_a", "oblique_b", "oblique_c"]
        else:
            view_names = ["top", "front", "right"]

        frames_pc, frames_feat, frame_valid, row_labels = [], [], [], []

        if not self.memory_bank.first_frame():
            frames_pc.append(self.memory_bank.anchor_pc)
            frames_feat.append(self.memory_bank.anchor_img_feat)
            frame_valid.append(True)
        else:
            frames_pc.append(pc_world_for_bank)
            frames_feat.append(img_feat_for_bank)
            frame_valid.append(False)
        row_labels.append("anchor (frame 0)")

        for k, slot in enumerate(self.memory_bank.ordered_slots()):
            if slot["pc"] is not None:
                frames_pc.append(slot["pc"])
                frames_feat.append(slot["feat"])
                frame_valid.append(True)
            else:
                frames_pc.append(pc_world_for_bank)
                frames_feat.append(img_feat_for_bank)
                frame_valid.append(False)
            row_labels.append(f"mem slot={k}")

        frames_pc.append(pc_world_for_bank)
        frames_feat.append(img_feat_for_bank)
        frame_valid.append(True)
        row_labels.append("current")

        rendered_rgbs = []
        for f_pc, f_feat, valid in zip(frames_pc, frames_feat, frame_valid):
            pc_cube = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in f_pc
            ]
            if getattr(self, "align_real_frame", False):
                pc_cube = [_pc.clone() for _pc in pc_cube]
                for _pc in pc_cube:
                    _pc[..., 0:2] = -_pc[..., 0:2]
            img = self._net_mod.render(
                pc=pc_cube, img_feat=f_feat,
                img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
            )
            rgb = img[:, :, 3:6, :, :].clamp(0, 1)
            if not valid:
                rgb = torch.zeros_like(rgb)
            rendered_rgbs.append(rgb[0])

        panel = torch.stack(rendered_rgbs, dim=0)
        panel = panel.permute(0, 1, 3, 4, 2).contiguous()
        panel = (panel.cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        _save_memory_panel(
            panel,
            os.path.join(save_dir, "memory_grid.png"),
            view_names=view_names,
            step_label=step_label,
            row_labels=row_labels,
        )

    def _update_shared(
        self,
        obs,
        pcd,
        action_trans_con,
        action_rot,
        action_grip,
        action_ignore_collisions,
        action_gripper_pose,
        replay_sample: dict,
        backprop: bool,
        reset_log: bool,
        memory_inputs: dict = None,
    ) -> dict:
        """Shared train step after bench-specific data extraction.

        Callers (`update` / `update_gembench`) pre-extract:
          obs, pcd                 — from the bench-specific preprocessor
          action_trans_con (bs, 3) — xyz
          action_rot       (bs, 4) — quaternion xyzw (torch.Tensor)
          action_grip      (bs,) int — 0 closed / 1 open
          action_ignore_collisions (bs, 1) int — 0 / 1
          action_gripper_pose (bs, >=7) — passed through to apply_se3_aug_con
        """
        return_out = {}
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)

            # SE3 augmentation. With memory enabled the SAME transform must hit anchor / history PCs, or the
            # fixed-camera per-pixel correspondence breaks — hence ``apply_se3_aug_con(extra_pcds=...)``.
            do_se3_aug = self._transform_augmentation and backprop
            extras_for_aug = None
            if do_se3_aug and memory_inputs is not None:
                extras_for_aug = (
                    [memory_inputs["anchor_pc_batched"]]
                    + list(memory_inputs["hist_pc_batched"])
                )
            if do_se3_aug:
                if extras_for_aug is not None:
                    # Visual KV PCs (anchor + history) need the same SE3 to stay co-rotated with current.
                    (action_trans_con, action_rot, pc,
                     perturbed_extras) = apply_se3_aug_con(
                        pcd=pc,
                        action_gripper_pose=action_gripper_pose,
                        bounds=torch.tensor(self.scene_bounds),
                        trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                        rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                        extra_pcds=extras_for_aug,
                    )
                    memory_inputs["anchor_pc_batched"] = perturbed_extras[0]
                    memory_inputs["hist_pc_batched"] = perturbed_extras[1:]
                else:
                    action_trans_con, action_rot, pc = apply_se3_aug_con(
                        pcd=pc,
                        action_gripper_pose=action_gripper_pose,
                        bounds=torch.tensor(self.scene_bounds),
                        trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                        rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                    )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            # align_real_frame is False for RLBench / GemBench; real/train.py sets it True for the robot base frame.
            if self.align_real_frame:
                _Rz180 = Rotation.from_euler('z', 180, degrees=True)
                for i in range(len(action_rot)):
                    action_rot[i] = (_Rz180 * Rotation.from_quat(action_rot[i])).as_quat()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)
            wpt_local = torch.cat(wpt_local, axis=0)

            # Finalize memory_inputs while ``pc`` is still post-bound: batched anchor / hist -> move_pc_in_bound
            # -> per-sample place_pc_in_cube, so they land in the same cube as the current frame.
            memory_inputs = self._finalize_memory_inputs(memory_inputs, pc)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            if self.align_real_frame:
                wpt_local[..., 0:2] = -wpt_local[..., 0:2]
                for _pc in pc:
                    _pc[..., 0:2] = -_pc[..., 0:2]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            img_aug = self.img_aug if (backprop and self.img_aug != 0) else 0
            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,      # (bs, 2)
            action_collision_one_hot, # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        rot_x_y = None
        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"],
            memory_inputs=memory_inputs,
        )

        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )
        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )

        loss_log = {}
        if backprop:
            # ---- Heatmap loss ----
            #   use_modified_focal_loss=False (default): action_trans is softmax-normalized over (h*w), and
            #     soft-label CE reproduces the original BridgeVLA loss exactly.
            #   True: peak-1 Gaussian + per-pixel sigmoid focal loss, mean over (bs, nc). stage_two doubles nc.
            if self.use_modified_focal_loss:
                bs_qt, hw_qt, nc_qt = q_trans.shape
                trans_logits = q_trans.transpose(1, 2).reshape(bs_qt, nc_qt, h, w)
                trans_gt = action_trans.transpose(1, 2).reshape(bs_qt, nc_qt, h, w)
                per_map = modified_focal_loss_per_heatmap(
                    trans_logits, trans_gt
                )  # (bs, nc_qt)
                num_img = self._net_mod.num_img
                if self.stage_two:
                    assert nc_qt == 2 * num_img, (nc_qt, num_img)
                    s1_per_view = per_map[:, :num_img]
                    s2_per_view = per_map[:, num_img:]
                    s1_trans_loss = s1_per_view.mean()
                    s2_trans_loss = s2_per_view.mean()
                    trans_loss = s1_trans_loss + s2_trans_loss
                    loss_log["trans_loss_s1"] = float(s1_trans_loss.item())
                    loss_log["trans_loss_s2"] = float(s2_trans_loss.item())
                else:
                    assert nc_qt == num_img, (nc_qt, num_img)
                    trans_loss = per_map.mean()
                    loss_log["trans_loss_s1"] = float(trans_loss.item())
                    loss_log["trans_loss_s2"] = 0.0
            else:
                # CE over (h*w) -> per-view loss (bs, nc); with stage_two, nc = 2*num_img laid out [s1 | s2].
                ce_per_view = self._cross_entropy_loss(q_trans, action_trans)
                trans_loss = ce_per_view.mean()
                num_img = self._net_mod.num_img
                if self.stage_two:
                    nc_total = ce_per_view.shape[-1]
                    assert nc_total == 2 * num_img, (nc_total, num_img)
                    s1_trans_loss = ce_per_view[:, :num_img].mean()
                    s2_trans_loss = ce_per_view[:, num_img:].mean()
                    loss_log["trans_loss_s1"] = float(s1_trans_loss.item())
                    loss_log["trans_loss_s2"] = float(s2_trans_loss.item())
                else:
                    loss_log["trans_loss_s1"] = float(trans_loss.item())
                    loss_log["trans_loss_s2"] = 0.0
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            rot_loss_6d = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                if self.rot_6d:
                    # 6D regression: Frobenius² to GT rotation matrix.
                    R_pred = aug_utils.rotation_6d_to_matrix(rot_q)
                    R_gt = self._gt_rotmat_from_quat(action_rot, rot_q.device)
                    rot_loss_6d = ((R_pred - R_gt) ** 2).sum(dim=(-1, -2)).mean()
                else:
                    rot_loss_x = self._cross_entropy_loss(
                        rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes],
                        action_rot_x_one_hot.argmax(-1),
                    ).mean()
                    rot_loss_y = self._cross_entropy_loss(
                        rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes],
                        action_rot_y_one_hot.argmax(-1),
                    ).mean()
                    rot_loss_z = self._cross_entropy_loss(
                        rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes],
                        action_rot_z_one_hot.argmax(-1),
                    ).mean()
                grip_loss = self._cross_entropy_loss(
                    grip_q, action_grip_one_hot.argmax(-1),
                ).mean()
                # Collision CE only when the bench has a real label; otherwise those logits get no gradient.
                if self.predict_collision:
                    collision_loss = self._cross_entropy_loss(
                        collision_q, action_collision_one_hot.argmax(-1),
                    ).mean()

            total_loss = (
                trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + rot_loss_6d
                + grip_loss + collision_loss
            )

            # ---- Keyframe discriminator BCE (mirrors _update_shared_dual) ----
            # Supervises out["mem_logit"] (post-memory stage-1 tokens) against the GT mem_label
            # (is-subtask-boundary); a no-op when mem_logit or mem_label is absent.
            mem_logit = out.get("mem_logit", None)
            if mem_logit is not None and "mem_label" in replay_sample:
                _disc_cfg = (getattr(self._net_mod, "memory_cfg", {}) or {}).get(
                    "discriminator", {}) or {}
                _lam = float(_disc_cfg.get("loss_weight", 1.0))
                _pw = float(_disc_cfg.get("pos_weight", 1.0))
                mem_label = replay_sample["mem_label"]
                if not torch.is_tensor(mem_label):
                    mem_label = torch.as_tensor(mem_label)
                mem_logit = mem_logit.reshape(-1)
                mem_label = mem_label.to(
                    mem_logit.device, dtype=mem_logit.dtype).reshape(-1)
                pos_weight = torch.as_tensor(
                    _pw, device=mem_logit.device, dtype=mem_logit.dtype)
                disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    mem_logit, mem_label, pos_weight=pos_weight,
                )
                total_loss = total_loss + _lam * disc_loss
                with torch.no_grad():
                    pred = (torch.sigmoid(mem_logit) > 0.5).float()
                    loss_log["disc_loss"] = float(disc_loss.item())
                    loss_log["disc_acc"] = float((pred == mem_label).float().mean().item())
                    loss_log["disc_gt_pos_rate"] = float(mem_label.mean().item())
                    loss_log["disc_pred_pos_rate"] = float(pred.mean().item())

            self._optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self._global_step += 1
            lr_scale = self._get_warmup_lr_scale()
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._lr * lr_scale
            self._optimizer.step()

            loss_log.update({
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": _loss_scalar(rot_loss_x),
                "rot_loss_y": _loss_scalar(rot_loss_y),
                "rot_loss_z": _loss_scalar(rot_loss_z),
                "rot_loss_6d": _loss_scalar(rot_loss_6d),
                "grip_loss": _loss_scalar(grip_loss),
                "lr": self._optimizer.param_groups[0]["lr"],
            })
            # Omit the key entirely when collision isn't supervised, so a flat-zero curve doesn't read as "converged".
            if self.predict_collision:
                loss_log["collision_loss"] = _loss_scalar(collision_loss)
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out

    def update(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
    ) -> dict:
        """RLBench (PerAct replay) train step. Thin adapter over `_update_shared`."""
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)

        action_rot_grip = replay_sample["rot_grip_action_indicies"][:, -1].int()   # (b, 4)
        action_ignore_collisions = None
        if self.predict_collision:
            assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
            action_ignore_collisions = replay_sample["ignore_collisions"][:, -1].int()  # (b, 1)
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]                 # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]                             # (b, 3)
        action_rot = action_gripper_pose[:, 3:7]                                   # (b, 4), xyzw
        action_grip = action_rot_grip[:, -1]                                       # (b,)

        obs, pcd = rlbench_utils._preprocess_inputs(replay_sample, self.cameras)
        return self._update_shared(
            obs=obs, pcd=pcd,
            action_trans_con=action_trans_con,
            action_rot=action_rot,
            action_grip=action_grip,
            action_ignore_collisions=action_ignore_collisions,
            action_gripper_pose=action_gripper_pose,
            replay_sample=replay_sample,
            backprop=backprop, reset_log=reset_log,
        )

    def update_gembench(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
    ) -> dict:
        """GemBench (3DLoTus LMDB) train step. Thin adapter over `_update_shared`."""
        action_gripper_pose = replay_sample["gripper_pose"]                               # (b, 8)
        # predict_collision=False (GemBench / memoryBench): those datasets no longer emit "ignore_collisions".
        action_ignore_collisions = None
        if self.predict_collision:
            action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int()  # (b, 1)
        action_trans_con = action_gripper_pose[:, 0:3]                                    # (b, 3)
        action_rot = action_gripper_pose[:, 3:7]                                          # (b, 4), xyzw
        # Openness is float in the raw action but keyframe data only emits 0 / 1; guard against silent .int() truncation.
        action_grip_float = action_gripper_pose[:, -1]
        assert torch.all((action_grip_float == 0) | (action_grip_float == 1)), (
            f"gembench action_grip must be in {{0, 1}}; got {action_grip_float.tolist()}"
        )
        action_grip = action_grip_float.int()                                             # (b,)

        obs, pcd = gembench_utils._preprocess_inputs_gembench(replay_sample, cameras)

        memory_inputs = None
        if self.memory_enabled:
            _net_mem_cfg = getattr(self._net_mod, "memory_cfg", {}) or {}
            K = int(_net_mem_cfg.get("k_temporal", 2))
            memory_inputs = self._build_memory_inputs_from_replay(
                replay_sample, cameras, K=K,
            )

        return self._update_shared(
            obs=obs, pcd=pcd,
            action_trans_con=action_trans_con,
            action_rot=action_rot,
            action_grip=action_grip,
            action_ignore_collisions=action_ignore_collisions,
            action_gripper_pose=action_gripper_pose,
            replay_sample=replay_sample,
            backprop=backprop, reset_log=reset_log,
            memory_inputs=memory_inputs,
        )

    def update_rmbench(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        cameras=("head", "front", "left", "right"),
    ) -> dict:
        """RMBench (dual-arm aloha-agilex) train step.

        ``replay_sample`` carries, per arm, the GT TCP-center action::

            left_action / right_action : (b, 8) = [xyz(3), quat_xyzw(4), grip(1)]

        where the TCP center is ``endpose_pos + R(endpose_quat)·[0.12,0,0]``
        and grip ∈ {0, 1}. The shared scene point cloud feeds mvt1 once; each
        arm zooms to its own waypoint at mvt2. Collision is NOT predicted
        (RMBench take_action('ee') has no collision flag).
        """
        arm_poses = [replay_sample["left_action"], replay_sample["right_action"]]
        return self._update_shared_dual(
            arm_poses=arm_poses,
            replay_sample=replay_sample,
            cameras=list(cameras),
            backprop=backprop,
            reset_log=reset_log,
        )

    def _update_shared_dual(
        self,
        arm_poses,
        replay_sample: dict,
        cameras,
        backprop: bool,
        reset_log: bool,
    ) -> dict:
        """Dual-arm train step. Mirrors ``_update_shared`` but:
          * the scene point cloud is shared (rendered once at mvt1);
          * SE3 augmentation co-transforms BOTH arms' GT poses with the
            same random transform as the shared cloud (and memory PCs);
          * the network runs the dual-arm forward (``out["per_arm"]``);
          * loss = Σ_arm (trans + rot_x + rot_y + rot_z + grip); no collision.

        Assumes ``place_with_mean=False`` (RMBench config), so the mvt1-cube
        projection is arm-independent: ``pc`` / ``rev_trans`` and the cube
        transform are shared; only the per-arm GT waypoint differs.
        """
        return_out = {}
        n_arms = len(arm_poses)

        obs, pcd = gembench_utils._preprocess_inputs_gembench(replay_sample, cameras)

        # Per-arm GT decomposition (xyzw).
        arm_trans_con = [p[:, 0:3] for p in arm_poses]                # list (bs,3)
        arm_rot = [p[:, 3:7] for p in arm_poses]                      # list (bs,4) xyzw
        # Gripper -> binary {0=closed, 1=open}. Threshold 0.9 (matches the dataset) keeps a gripper holding an object closed.
        arm_grip = [(p[:, -1] > 0.9).int() for p in arm_poses]

        memory_inputs = None
        if self.memory_enabled:
            _net_mem_cfg = getattr(self._net_mod, "memory_cfg", {}) or {}
            K = int(_net_mem_cfg.get("k_temporal", 2))
            memory_inputs = self._build_memory_inputs_from_replay(
                replay_sample, cameras, K=K,
            )

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)

            # SE3 aug: one transform across cloud + arm0 (pivot) + arm1 + memory PCs.
            do_se3_aug = self._transform_augmentation and backprop
            if do_se3_aug:
                extras_for_aug = None
                if memory_inputs is not None:
                    extras_for_aug = (
                        [memory_inputs["anchor_pc_batched"]]
                        + list(memory_inputs["hist_pc_batched"])
                    )
                extra_action_poses = arm_poses[1:]  # arm0 is the primary
                ret = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=arm_poses[0],
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                    extra_pcds=extras_for_aug,
                    extra_action_poses=extra_action_poses,
                )
                # Unpack: (trans0, quat0, pc[, extras][, extra_poses]).
                arm0_trans, arm0_quat, pc = ret[0], ret[1], ret[2]
                idx = 3
                if extras_for_aug is not None:
                    perturbed_extras = ret[idx]; idx += 1
                    memory_inputs["anchor_pc_batched"] = perturbed_extras[0]
                    memory_inputs["hist_pc_batched"] = perturbed_extras[1:]
                perturbed_extra_poses = ret[idx]
                arm_trans_con = [torch.tensor(arm0_trans).to(pc.device)]
                arm_rot = [torch.tensor(arm0_quat).to(pc.device)]
                for (t_np, q_np) in perturbed_extra_poses:
                    arm_trans_con.append(torch.tensor(t_np).to(pc.device))
                    arm_rot.append(torch.tensor(q_np).to(pc.device))

            # Normalize each arm's quaternion (xyzw, hemisphere-fixed).
            arm_rot_np = []
            for _rot in arm_rot:
                _rot = _rot.cpu().numpy() if isinstance(_rot, torch.Tensor) else _rot
                if self.align_real_frame:
                    _Rz180 = Rotation.from_euler('z', 180, degrees=True)
                    for i in range(len(_rot)):
                        _rot[i] = (_Rz180 * Rotation.from_quat(_rot[i])).as_quat()
                for i, _r in enumerate(_rot):
                    _r = aug_utils.normalize_quaternion(_r)
                    if _r[-1] < 0:
                        _r = -_r
                    _rot[i] = _r
                arm_rot_np.append(_rot)

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )

            # Cube projection is arm-independent (place_with_mean=False): build pc / rev_trans once, reuse per arm.
            assert not self._place_with_mean, (
                "RMBench dual-arm assumes place_with_mean=False"
            )
            pc_cube = []
            rev_trans = []
            for _pc in pc:
                a, b = mvt_utils.place_pc_in_cube(
                    _pc, with_mean_or_bounds=False, scene_bounds=self.scene_bounds,
                )
                pc_cube.append(a)
                rev_trans.append(b)

            # Per-arm wpt_local in the shared cube (via place_pc_in_cube's app_pc path).
            wpt_local_arms = []
            for a_idx in range(n_arms):
                wl = []
                for _pc, _wpt in zip(pc, [x[:3] for x in arm_trans_con[a_idx]]):
                    a_cube, _ = mvt_utils.place_pc_in_cube(
                        _pc, _wpt,
                        with_mean_or_bounds=False, scene_bounds=self.scene_bounds,
                    )
                    wl.append(a_cube.unsqueeze(0))
                wpt_local_arms.append(torch.cat(wl, axis=0))

            # Finalize memory (anchor/hist into shared cube) BEFORE pc clobber.
            memory_inputs = self._finalize_memory_inputs(memory_inputs, pc)

            pc = pc_cube
            if self.align_real_frame:
                for a_idx in range(n_arms):
                    wpt_local_arms[a_idx][..., 0:2] = -wpt_local_arms[a_idx][..., 0:2]
                for _pc in pc:
                    _pc[..., 0:2] = -_pc[..., 0:2]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size
            img_aug = self.img_aug if (backprop and self.img_aug != 0) else 0
            dyn_cam_info = None

            # Per-arm one-hot expert actions + rot_x_y.
            per_arm_onehot = []
            rot_x_y_arms = []
            for a_idx in range(n_arms):
                (rx, ry, rz, grip_oh, coll_oh) = self._get_one_hot_expert_actions(
                    bs, arm_rot_np[a_idx], arm_grip[a_idx],
                    torch.zeros(bs, 1, dtype=torch.int, device=self._device),
                    device=self._device,
                )
                per_arm_onehot.append((rx, ry, rz, grip_oh, coll_oh))
                if self.rot_ver == 1:
                    rxy = torch.cat(
                        [rx.argmax(dim=-1, keepdim=True), ry.argmax(dim=-1, keepdim=True)],
                        dim=-1,
                    )
                    if self.rot_x_y_aug != 0:
                        rxy = rxy + torch.randint(
                            -self.rot_x_y_aug, self.rot_x_y_aug, size=rxy.shape
                        ).to(rxy.device)
                        rxy = rxy % self._num_rotation_classes
                    rot_x_y_arms.append(rxy)
                else:
                    rot_x_y_arms.append(None)

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=[wl if self._network.training else None for wl in wpt_local_arms],
            rot_x_y=rot_x_y_arms if self.rot_ver == 1 else [None] * n_arms,
            language_goal=replay_sample["lang_goal"],
            memory_inputs=memory_inputs,
        )

        loss_log = {}
        total_loss = 0.0
        for a_idx in range(n_arms):
            out_arm = out["per_arm"][a_idx]
            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
                out_arm, dims=(bs, nc, h, w)
            )
            action_trans = self.get_action_trans(
                wpt_local_arms[a_idx], pts, out_arm, dyn_cam_info, dims=(bs, nc, h, w)
            )
            if not backprop:
                continue

            (rx, ry, rz, grip_oh, _coll) = per_arm_onehot[a_idx]
            # Heatmap (soft-label CE; RMBench uses use_modified_focal_loss=False).
            ce_per_view = self._cross_entropy_loss(q_trans, action_trans)
            trans_loss = ce_per_view.mean()
            num_img = self._net_mod.num_img
            if self.stage_two:
                s1 = ce_per_view[:, :num_img].mean()
                s2 = ce_per_view[:, num_img:].mean()
                loss_log[f"trans_loss_s1_arm{a_idx}"] = float(s1.item())
                loss_log[f"trans_loss_s2_arm{a_idx}"] = float(s2.item())

            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            rot_loss_6d = 0.0
            grip_loss = 0.0
            if self.add_rgc_loss:
                if self.rot_6d:
                    # 6D regression: Gram-Schmidt -> R_pred vs R_gt, Frobenius² — a stable monotone surrogate
                    # of the geodesic angle (Zhou et al., CVPR 2019), with no discretization.
                    R_pred = aug_utils.rotation_6d_to_matrix(rot_q)        # (bs,3,3)
                    R_gt = self._gt_rotmat_from_quat(
                        arm_rot_np[a_idx], rot_q.device)                   # (bs,3,3)
                    rot_loss_6d = ((R_pred - R_gt) ** 2).sum(dim=(-1, -2)).mean()
                    with torch.no_grad():
                        geo_deg = (aug_utils.geodesic_distance(R_pred, R_gt).mean()
                                   * 180.0 / np.pi)
                    loss_log[f"rot_geo_deg_arm{a_idx}"] = float(geo_deg.item())
                else:
                    R = self._num_rotation_classes
                    rot_loss_x = self._cross_entropy_loss(
                        rot_q[:, 0 * R:1 * R], rx.argmax(-1)).mean()
                    rot_loss_y = self._cross_entropy_loss(
                        rot_q[:, 1 * R:2 * R], ry.argmax(-1)).mean()
                    rot_loss_z = self._cross_entropy_loss(
                        rot_q[:, 2 * R:3 * R], rz.argmax(-1)).mean()
                grip_loss = self._cross_entropy_loss(
                    grip_q, grip_oh.argmax(-1)).mean()
            # NOTE: collision_q is computed but NOT supervised for RMBench.

            arm_loss = (trans_loss + rot_loss_x + rot_loss_y + rot_loss_z
                        + rot_loss_6d + grip_loss)
            total_loss = total_loss + arm_loss
            loss_log[f"trans_loss_arm{a_idx}"] = float(trans_loss.item())
            loss_log[f"grip_loss_arm{a_idx}"] = float(
                grip_loss.item() if torch.is_tensor(grip_loss) else grip_loss)
            loss_log[f"rot_loss_arm{a_idx}"] = float(
                sum(x.item() if torch.is_tensor(x) else x
                    for x in (rot_loss_x, rot_loss_y, rot_loss_z, rot_loss_6d)))

        # ---- Keyframe discriminator BCE (scene-level, shared across arms) ----
        # Supervises out["mem_logit"] against the GT mem_label; ~15% positives, handled via pos_weight.
        # Backprops jointly into the memory blocks + PaliGemma through x_heads.
        mem_logit = out.get("mem_logit", None)
        if backprop and mem_logit is not None and "mem_label" in replay_sample:
            _disc_cfg = (getattr(self._net_mod, "memory_cfg", {}) or {}).get(
                "discriminator", {}) or {}
            _lam = float(_disc_cfg.get("loss_weight", 1.0))
            _pw = float(_disc_cfg.get("pos_weight", 1.0))
            mem_label = replay_sample["mem_label"]
            if not torch.is_tensor(mem_label):
                mem_label = torch.as_tensor(mem_label)
            mem_logit = mem_logit.reshape(-1)
            mem_label = mem_label.to(
                mem_logit.device, dtype=mem_logit.dtype).reshape(-1)
            pos_weight = torch.as_tensor(
                _pw, device=mem_logit.device, dtype=mem_logit.dtype)
            disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                mem_logit, mem_label, pos_weight=pos_weight,
            )
            total_loss = total_loss + _lam * disc_loss
            with torch.no_grad():
                pred = (torch.sigmoid(mem_logit) > 0.5).float()
                loss_log["disc_loss"] = float(disc_loss.item())
                loss_log["disc_acc"] = float((pred == mem_label).float().mean().item())
                loss_log["disc_gt_pos_rate"] = float(mem_label.mean().item())
                loss_log["disc_pred_pos_rate"] = float(pred.mean().item())

        if backprop:
            self._optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self._global_step += 1
            lr_scale = self._get_warmup_lr_scale()
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._lr * lr_scale
            self._optimizer.step()
            loss_log["total_loss"] = float(total_loss.item())
            loss_log["lr"] = self._optimizer.param_groups[0]["lr"]
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out


    def _save_pred_views(self, out, save_dir, step_id, sample_idx=0, heatmap=None):
        """Save Row1=rendered top/front/right RGB (with view weight info),
        Row2=heatmap overlay on the same RGB images."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        os.makedirs(save_dir, exist_ok=True)

        rendered_img = out["mvt1_ori_img"]
        rgb = rendered_img[sample_idx, :, 3:6].cpu().float()

        num_views = rgb.shape[0]
        # In oblique mode the renderer's 3 views are magic-angle oblique views, so the captions are renamed accordingly.
        _renderer = getattr(self._net_mod, "renderer", None)
        if getattr(_renderer, "oblique_views", False):
            view_names = ["oblique_a", "oblique_b", "oblique_c"][:num_views]
        else:
            view_names = ["top", "front", "right"][:num_views]
        rgb_np = np.clip(rgb.numpy().transpose(0, 2, 3, 1), 0, 1)

        # Shared heatmap color scale across views.
        hm_vmin = float(heatmap.min()) if heatmap is not None else 0.0
        hm_vmax = float(heatmap.max()) if heatmap is not None else 1.0

        fig, axes = plt.subplots(2, num_views, figsize=(4 * num_views, 8))
        for v in range(num_views):
            orig_uint8 = (np.clip(rgb_np[v], 0, 1) * 255).astype(np.uint8)

            axes[0, v].imshow(orig_uint8)
            axes[0, v].set_title(view_names[v])
            axes[0, v].axis("off")

            if heatmap is not None:
                hm = heatmap[v].astype(np.float64)
                if hm_vmax - hm_vmin > 1e-8:
                    hm_norm = np.clip((hm - hm_vmin) / (hm_vmax - hm_vmin), 0, 1)
                else:
                    hm_norm = np.zeros_like(hm)
                heatmap_rgb = (cm.jet(hm_norm) * 255).astype(np.uint8)[..., :3]
                blended = (0.5 * orig_uint8.astype(np.float64)
                           + 0.5 * heatmap_rgb.astype(np.float64))
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                axes[1, v].imshow(blended)
            else:
                axes[1, v].imshow(orig_uint8)
            axes[1, v].set_title(f"{view_names[v]} + overlay")
            axes[1, v].axis("off")

        plt.suptitle(f"Step {step_id}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pred_views_{step_id:06d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def _stage2_zoom_half_extent(self):
        """Half-extent (in mvt1-cube coords) of the cube ACTUALLY visible in
        the stage-2 zoom render. ``trans_pc`` scales by ``st_sca``, and the
        renderer's orthographic viewport spans ±``renderer_img_sizes_w/2``
        world units (NOT ±1; e.g. img_sizes_w=0.8 -> ±0.4). So the visible
        cube is |pc - wpt_local1| <= (img_sizes_w/2) / st_sca. With the old
        default img_sizes_w=2.0 this reduces to the previous 1/st_sca."""
        _sca = float(getattr(self._net_mod, "st_sca", 4.0))
        _sizes_w = getattr(self._net_mod, "renderer_img_sizes_w", None) or [2.0, 2.0]
        return (min(float(s) for s in _sizes_w) / 2.0) / _sca

    def act(
        self, step: int, observation: dict,deterministic=True,visualize=False,visualize_save_dir="", return_gembench_action=False,
    ) -> ActResult:
        language_goal =observation["language_goal"]
        obs, pcd = rlbench_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )
        pc_ori = pc[0].clone()
        img_feat_ori=img_feat[0].clone()
        # Capture the post-bound PC for the memory-bank push before cube projection clobbers ``pc``.
        pc_world_for_bank = [_pc.clone() for _pc in pc]
        img_feat_for_bank = [_f.clone() for _f in img_feat]

        # ---- Episodic memory: build memory_inputs from bank state ----
        # Stage-1 cameras are fixed and the memory KV path is detached, so anchor / history tokens are cached
        # and reused verbatim; stage 2's anchor is re-rendered each step, hence the raw PCs kept in the bank.
        memory_inputs = None
        if self.memory_enabled:
            _net_mem_cfg = getattr(self._net_mod, "memory_cfg", {}) or {}
            K = int(_net_mem_cfg.get("k_temporal", 2))
            bs_act = len(pc)
            first_frame = self.memory_bank.first_frame()

            if first_frame:
                # Episode start: the bank's anchor slot is empty, so hand MVT the *current* PC as a placeholder
                # plus a zero ``anchor_mask`` — matching training semantics at step_idx == 0. The spatial
                # block's cross-attention is zeroed while its residuals stay active, so train and eval agree.
                anchor_pc_world = [_p.clone() for _p in pc_world_for_bank]
                anchor_feat = [_f.clone() for _f in img_feat_for_bank]
                anchor_mask_t = torch.zeros(bs_act, dtype=torch.bool, device=self._device)
                anchor_tokens_t = None
            else:
                # Later steps: cached anchor tokens are reused; the raw PC is kept for stage-2 zoom re-rendering.
                anchor_pc_world = list(self.memory_bank.anchor_pc)
                anchor_feat = list(self.memory_bank.anchor_img_feat)
                anchor_mask_t = torch.ones(bs_act, dtype=torch.bool, device=self._device)
                anchor_tokens_t = self.memory_bank.anchor_mvt1_tokens

            # History: stack cached tokens into (bs, K, vlm_dim, V, H_p, W_p), zero-padding empty slots (the mask gates them).
            n_hist = self.memory_bank.num_history()
            hist_tokens_t, hist_mask_t = stack_hist_tokens_from_bank(
                self.memory_bank, bs_act, K, self._device,
            )

            memory_inputs = {
                "anchor_pc": anchor_pc_world,
                "anchor_img_feat": anchor_feat,
                "anchor_mask": anchor_mask_t,
                "anchor_mvt1_tokens": anchor_tokens_t,
                # ``hist_pc`` / ``hist_img_feat`` are absent at eval — temporal memory is stage-1-only and cached.
                "hist_mask": hist_mask_t,
                "hist_mvt1_tokens": hist_tokens_t,
            }
            # Project the anchor's raw PC (used for stage-2 zoom) into current's cube; history has no raw PC at eval.
            memory_inputs = self._project_memory_inputs_to_cube(
                memory_inputs, pc,
            )

        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,  # no img augmentation while acting
            language_goal=language_goal,
            memory_inputs=memory_inputs,
        )
        if visualize:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )            
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        # ---- Stage-2 zoom-empty translation fallback (eval-only) ----
        # mvt2 only sees a cube of half-extent (img_sizes_w/2)/st_sca around stage 1's waypoint (see
        # ``_stage2_zoom_half_extent``). When that cube is near-empty its heatmap has no visual anchor, and
        # ``rev_trans`` multiplies the argmax error by ``st_sca``, giving a far drift. Mitigation: with fewer
        # than ``_STAGE2_ZOOM_MIN_PTS`` points inside, take stage 1's prediction (the zoom-cube center) for
        # translation only — rotation / gripper / collision still come from mvt2. Training is untouched.
        _STAGE2_ZOOM_MIN_PTS = 5000
        zoom_pt_counts = None
        zoom_half_extent = None
        _force_stage2_center = _rmbench_stage2_force_center()
        _sparsepc_force_center = _rmbench_stage2_sparsepc_force_center()
        if self.stage_two and "wpt_local1" in out:
            zoom_half_extent = self._stage2_zoom_half_extent()
            _wpt_local1 = out["wpt_local1"]  # (bs, 3) in mvt1 cube coords
            zoom_pt_counts = []
            for _b in range(bs):
                _delta = pc[_b] - _wpt_local1[_b].unsqueeze(0)
                _in_zoom = (_delta.abs() <= zoom_half_extent).all(dim=-1)
                _cnt = int(_in_zoom.sum().item())
                zoom_pt_counts.append(_cnt)
                if _force_stage2_center or (_sparsepc_force_center and _cnt < _STAGE2_ZOOM_MIN_PTS):
                    # rev_trans[_b] maps mvt1 cube coords -> world coords.
                    pred_wpt[_b] = rev_trans[_b](_wpt_local1[_b])

        if visualize:
            print("Visualizing")
            save_dir = visualize_save_dir
            os.makedirs(save_dir, exist_ok=True)
            # Remember this episode's viz root + step labels for finalize_eval_viz(); a new save dir means a new episode.
            _step_label = f"step{int(step)}"
            if self._eval_viz_episode_dir != visualize_save_dir:
                self._eval_viz_episode_dir = visualize_save_dir
                self._eval_viz_step_labels = []
            if _step_label not in self._eval_viz_step_labels:
                self._eval_viz_step_labels.append(_step_label)
            try:
                self._eval_viz_instruction = language_goal[0][0][0]
            except Exception:
                self._eval_viz_instruction = None
            save_dir = os.path.join(save_dir, _step_label)
            os.makedirs(save_dir, exist_ok=True)

            # Reuse training-time visualize.py's helper so per-view layout, panel split, colorbar scale and
            # titles stay in sync. At eval there is no GT, so ``action_trans=None`` emits a logits-only panel.
            for _stage in ("mvt1", "mvt2"):
                _save_stage(
                    out=out,
                    q_trans=q_trans,
                    action_trans=None,
                    sample_idx=0,
                    sample_dir=save_dir,
                    stage=_stage,
                    nc=nc,
                    h=h,
                    w=w,
                    use_modified_focal_loss=self.use_modified_focal_loss,
                )

            # Stage-2 zoom-cube point counts (debug), written only when visualize=True. The mvt2 heatmap PNGs
            # still show mvt2's RAW argmax even when pred_wpt is overridden below.
            if zoom_pt_counts is not None:
                _mvt2_dir = os.path.join(save_dir, "mvt2")
                os.makedirs(_mvt2_dir, exist_ok=True)
                with open(os.path.join(_mvt2_dir, "zoom_pt_count.txt"), "w") as _f:
                    _f.write(f"st_sca={float(getattr(self._net_mod, 'st_sca', 4.0))}\n")
                    _f.write(
                        "renderer_img_sizes_w="
                        f"{getattr(self._net_mod, 'renderer_img_sizes_w', [2.0, 2.0])}\n"
                    )
                    _f.write(f"zoom_half_extent_cube={zoom_half_extent}\n")
                    _f.write(f"threshold={_STAGE2_ZOOM_MIN_PTS}\n")
                    for _b, _cnt in enumerate(zoom_pt_counts):
                        if _force_stage2_center:
                            _tag = "FORCE_CENTER"
                        else:
                            if _sparsepc_force_center:
                                _tag = ("FALLBACK_TO_STAGE1"
                                        if _cnt < _STAGE2_ZOOM_MIN_PTS else "USE_MVT2")
                            else:
                                _tag = "USE_MVT2"
                        _f.write(f"sample{_b} count={_cnt} {_tag}\n")

            # Stage-2 memory grid (anchor + current at the zoomed cameras); the anchor is gated black at step 0.
            if "mvt2_anchor_ori_img" in out and "mvt2_ori_img" in out:
                _renderer = getattr(self._net_mod, "renderer", None)
                if getattr(_renderer, "oblique_views", False):
                    _view_names = ["oblique_a", "oblique_b", "oblique_c"]
                else:
                    _view_names = ["top", "front", "right"]
                _mvt2_raw = q_trans[0, :, nc:2 * nc].clone().view(h, w, nc).float()
                if self.use_modified_focal_loss:
                    _mvt2_hm = torch.sigmoid(_mvt2_raw).permute(2, 0, 1).contiguous()
                else:
                    _flat = _mvt2_raw.permute(2, 0, 1).reshape(nc, h * w)
                    _mvt2_hm = torch.softmax(_flat, dim=-1).view(nc, h, w)
                _anchor_valid = True
                if memory_inputs is not None and "anchor_mask" in memory_inputs:
                    _anchor_valid = bool(memory_inputs["anchor_mask"][0].item())
                # Anchor-vs-current comparison next to this step's mvt2 panels (step{N}/mvt2/anchor_memory_grid.png).
                _mvt2_dir = os.path.join(save_dir, "mvt2")
                os.makedirs(_mvt2_dir, exist_ok=True)
                _save_mvt2_memory_grid(
                    anchor_img_one=out["mvt2_anchor_ori_img"][0, :, 3:6],
                    current_img_one=out["mvt2_ori_img"][0, :, 3:6],
                    current_hm_one=_mvt2_hm,
                    anchor_valid=_anchor_valid,
                    save_path=os.path.join(_mvt2_dir, "anchor_memory_grid.png"),
                    view_names=_view_names,
                    step_label=f"(step {step})",
                )
                # The same anchor row, but one PNG per view under step{N}/anchor_frame/{view_name}.png.
                _save_anchor_frames(
                    anchor_img_one=out["mvt2_anchor_ori_img"][0, :, 3:6],
                    anchor_valid=_anchor_valid,
                    save_dir=os.path.join(save_dir, "anchor_frame"),
                    view_names=_view_names,
                )

            # The temporal memory panel (memory_grid.png) is deliberately not written here: re-rendering the
            # whole anchor + history bank per step is expensive, and only RMBench's act_rmbench needs it.

            # Heatmap for the pred_views grid: post-activation mvt1 stage of q_trans, (V, H, W). The activation
            # must match the training objective (sigmoid for focal loss, per-view spatial softmax otherwise).
            _mvt1_raw = q_trans[0, :, :nc].clone().view(h, w, nc).float()
            if self.use_modified_focal_loss:
                _mvt1_hm = torch.sigmoid(_mvt1_raw)
            else:
                _flat = _mvt1_raw.permute(2, 0, 1).reshape(nc, h * w)
                _mvt1_hm = torch.softmax(_flat, dim=-1).view(nc, h, w)
                _mvt1_hm = _mvt1_hm.permute(1, 2, 0).contiguous()
            hm_for_views = _mvt1_hm.cpu().numpy().transpose(2, 0, 1)
            self._save_pred_views(out, save_dir=save_dir, step_id=int(step),
                                  heatmap=hm_for_views)
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )

        # ---- Episodic memory: end-of-step update ----
        # Push this step's PRE-memory stage-1 tokens (detached, bf16) into the eval bank; raw PCs / img_feats are kept only for visualization.
        if self.memory_enabled:
            curr_tokens = out.get("mvt1_paligemma_tokens", None)
            if curr_tokens is not None:
                # Keep on GPU in bf16 — the bank holds at most K+1 tokens, a few MB at bs=1.
                curr_tokens = curr_tokens.detach()

            if self.memory_bank.first_frame():
                # Episode start: the current tokens ARE the anchor tokens, so no extra PaliGemma forward is
                # needed — the spatial block was already gated to identity by the zero ``anchor_mask``.
                self.memory_bank.set_anchor(
                    [_p.detach() for _p in pc_world_for_bank],
                    [_f.detach() for _f in img_feat_for_bank],
                    curr_tokens,
                )

            # Keyframe-discriminator admission gate (mirrors act_rmbench):
            #   discriminator present -> admit iff sigmoid(logit) > threshold;
            #   keyframe_gt without one (MemoryBench) -> never admit; sliding (legacy GemBench) -> admit all.
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

        if return_gembench_action:
            continuous_action = np.concatenate(
                    [
                        pred_wpt[0].cpu().numpy(),
                        pred_rot_quat[0],
                        pred_grip[0].cpu().numpy(),
                    ], -1
                )
            return continuous_action
        else:
            return ActResult(continuous_action)

    def finalize_eval_viz(self):
        """Stitch the current eval episode's per-step overlay tri-views into
        per-stage grids and clear the tracking state.

        Single-arm counterpart of RMBench's ``finalize_episode_viz`` RPC: the
        per-bench eval driver calls this once at the end of each episode
        (RLBench in ``eval.py``; GemBench / MemoryBench via the server
        ``/finalize`` route) so each episode gets ``grid_mvt1.png`` /
        ``grid_mvt2.png`` (rows=steps, cols=views) under its episode dir.
        No-op when overlay viz wasn't active this episode.
        """
        episode_dir = self._eval_viz_episode_dir
        step_labels = list(self._eval_viz_step_labels)
        instruction = self._eval_viz_instruction
        # Clear up-front so a stitch failure (or a missed finalize) can't leak this episode's steps into the next.
        self._eval_viz_episode_dir = None
        self._eval_viz_step_labels = []
        self._eval_viz_instruction = None
        if not episode_dir or not step_labels:
            return {"ok": True, "stitched": False}
        try:
            net = self._net_mod
            n_views = int(net.num_img)
            stages = ("mvt1", "mvt2") if self.stage_two else ("mvt1",)
            renderer = getattr(net, "renderer", None)
            if getattr(renderer, "oblique_views", False):
                view_names = ["oblique_a", "oblique_b", "oblique_c"]
            else:
                view_names = ["top", "front", "right"]
            stitch_episode_overlays(
                episode_dir, step_labels, stages=stages,
                n_views=n_views, view_names=view_names, arms=None,
                instruction=instruction,
            )
            print(f"[eval viz] stitched episode grids -> {episode_dir}")
            return {"ok": True, "stitched": True}
        except Exception as e:
            import traceback
            print(f"[eval viz] stitch failed for {episode_dir}: {e}")
            traceback.print_exc()
            return {"ok": False, "stitched": False}

    @torch.no_grad()
    def act_rmbench(self, observation: dict, visualize=False, visualize_save_dir="",
                    diag_save_dir=""):
        """Dual-arm eval step for RMBench.

        ``observation`` must already carry the bridgevla per-camera obs
        (``{cam}_rgb`` / ``{cam}_point_cloud`` as (1, 3, H, W) tensors) and
        ``language_goal`` (triple-wrapped). Returns, per arm, the predicted
        TCP-center pose + gripper::

            {"left": (wpt(3) np, quat_xyzw(4) np, grip float),
             "right": (...)}

        The caller (deploy_policy) converts xyzw→wxyz and assembles the
        16-D ee action. Memory bank: mvt1 anchor/history are shared (one
        bank); the per-arm mvt2 anchor zoom happens inside the network
        forward, so the bank push is identical to single-arm ``act()``.

        ``visualize=True`` additionally dumps per-arm pred heatmap overlays
        for this step under ``visualize_save_dir`` — REUSING the same ``out``
        already computed for the action (no extra forward) and the same
        ``_save_stage`` helper as training-time ``RMBench_vla/visualize.py``.
        Also writes ``memory_grid.png`` (shared mvt1 anchor + temporal memory
        + current) before the end-of-step bank push. At eval there is no GT, so ``action_trans=None`` emits pred-only
        panels. Mirrors the single-arm ``act(visualize=True)`` path, looped
        over ``out["per_arm"]`` like train's ``_viz_one_sample``.
        """
        language_goal = observation["language_goal"]
        obs, pcd = rlbench_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )
        pc_world_for_bank = [_pc.clone() for _pc in pc]
        img_feat_for_bank = [_f.clone() for _f in img_feat]

        # ---- Episodic memory inputs (shared across arms at mvt1). ----
        memory_inputs = None
        if self.memory_enabled:
            _net_mem_cfg = getattr(self._net_mod, "memory_cfg", {}) or {}
            K = int(_net_mem_cfg.get("k_temporal", 2))
            bs_act = len(pc)
            first_frame = self.memory_bank.first_frame()
            if first_frame:
                anchor_pc_world = [_p.clone() for _p in pc_world_for_bank]
                anchor_feat = [_f.clone() for _f in img_feat_for_bank]
                anchor_mask_t = torch.zeros(bs_act, dtype=torch.bool, device=self._device)
                anchor_tokens_t = None
            else:
                anchor_pc_world = list(self.memory_bank.anchor_pc)
                anchor_feat = list(self.memory_bank.anchor_img_feat)
                anchor_mask_t = torch.ones(bs_act, dtype=torch.bool, device=self._device)
                anchor_tokens_t = self.memory_bank.anchor_mvt1_tokens

            hist_tokens_t, hist_mask_t = stack_hist_tokens_from_bank(
                self.memory_bank, bs_act, K, self._device,
            )

            memory_inputs = {
                "anchor_pc": anchor_pc_world,
                "anchor_img_feat": anchor_feat,
                "anchor_mask": anchor_mask_t,
                "anchor_mvt1_tokens": anchor_tokens_t,
                "hist_mask": hist_mask_t,
                "hist_mvt1_tokens": hist_tokens_t,
            }
            memory_inputs = self._project_memory_inputs_to_cube(memory_inputs, pc)

        # Shared cube projection (place_with_mean=False); rev_trans_world maps mvt1-cube -> world for all arms.
        pc_new = []
        rev_trans_world = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans_world.append(b)
        pc = pc_new

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,
            language_goal=language_goal,
            memory_inputs=memory_inputs,
        )

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        n_arms = self._net_mod.num_arms

        result = {}
        arm_names = ["left", "right"][:n_arms]
        pred_grips_for_viz = []
        zoom_diag_entries = []
        _STAGE2_ZOOM_MIN_PTS = 5000
        _zoom_half_extent = None
        _st_sca = None
        _force_stage2_center = _rmbench_stage2_force_center()
        _sparsepc_force_center = _rmbench_stage2_sparsepc_force_center()
        if self.stage_two:
            _st_sca = float(getattr(self._net_mod, "st_sca", 4.0))
            # Cube actually visible in the mvt2 zoom render: the ortho viewport spans ±img_sizes_w/2, not ±1.
            _zoom_half_extent = self._stage2_zoom_half_extent()
        for a_idx in range(n_arms):
            out_arm = out["per_arm"][a_idx]
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out_arm, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )
            # get_pred maps mvt2-zoom -> mvt1-cube via out_arm["rev_trans"], then mvt1-cube -> world via the
            # shared rev_trans_world (arm-independent because place_with_mean=False).
            pred_wpt, pred_rot_quat, pred_grip, _pred_coll = self.get_pred(
                out_arm, rot_q, grip_q, collision_q, y_q,
                rev_trans_world, dyn_cam_info
            )

            # ---- Stage-2 zoom-empty translation fallback (eval-only, per arm) ----
            # As in single-arm act(): too few points in this arm's zoom cube -> translation falls back to the
            # zoom center (wpt_local1); rotation / gripper still come from stage 2.
            if (self.stage_two and "wpt_local1" in out_arm
                    and _zoom_half_extent is not None):
                _wpt_local1 = out_arm["wpt_local1"]
                for _b in range(bs):
                    _delta = pc[_b] - _wpt_local1[_b].unsqueeze(0)
                    _in_zoom = (_delta.abs() <= _zoom_half_extent).all(dim=-1)
                    _cnt = int(_in_zoom.sum().item())
                    _used_fallback = _force_stage2_center or (
                        _sparsepc_force_center and _cnt < _STAGE2_ZOOM_MIN_PTS
                    )
                    if _used_fallback:
                        pred_wpt[_b] = rev_trans_world[_b](_wpt_local1[_b])
                    if _b == 0:
                        zoom_diag_entries.append({
                            "arm": f"{arm_names[a_idx]}_arm",
                            "count": _cnt,
                            "used_fallback": _used_fallback,
                            "force_center": _force_stage2_center,
                        })

            pred_grip_val = int(pred_grip[0].cpu().numpy())
            result[arm_names[a_idx]] = (
                pred_wpt[0].cpu().numpy(),          # TCP center xyz (world)
                pred_rot_quat[0],                   # quat xyzw
                float(pred_grip_val),               # grip 0/1
            )
            if visualize or diag_save_dir:
                pred_grips_for_viz.append(pred_grip_val)

            # ---- Per-arm eval-time heatmap viz (reuses out, no re-forward) ----
            # Same plumbing as single-arm act(visualize=True); wrapped in try/except so viz never breaks acting.
            if visualize:
                try:
                    q_trans, _, _, _, _, _ = self.get_q(
                        out_arm, dims=(bs, nc, h, w),
                        only_pred=True, get_q_trans=True,
                    )
                    arm_tag = ["L", "R"][a_idx] if a_idx < 2 else f"arm{a_idx}"
                    arm_dir = os.path.join(visualize_save_dir, f"arm_{arm_tag}")
                    os.makedirs(arm_dir, exist_ok=True)
                    stages = ("mvt1", "mvt2") if self.stage_two else ("mvt1",)
                    for _stage in stages:
                        _save_stage(
                            out=out_arm, q_trans=q_trans, action_trans=None,
                            sample_idx=0, sample_dir=arm_dir, stage=_stage,
                            nc=nc, h=h, w=w,
                            use_modified_focal_loss=self.use_modified_focal_loss,
                        )

                    # Anchor-memory comparison for this arm's stage-2 zoom: anchor and current re-rendered at
                    # the SAME per-arm mvt2 cameras, so the two rows are pixel-comparable. Blacked out on frame 0.
                    if (self.stage_two
                            and "mvt2_anchor_ori_img" in out_arm
                            and "mvt2_ori_img" in out_arm):
                        _renderer = getattr(self._net_mod, "renderer", None)
                        if getattr(_renderer, "oblique_views", False):
                            _view_names = ["oblique_a", "oblique_b", "oblique_c"]
                        else:
                            _view_names = ["top", "front", "right"]
                        _mvt2_raw = q_trans[0, :, nc:2 * nc].clone().view(
                            h, w, nc
                        ).float()
                        if self.use_modified_focal_loss:
                            _mvt2_hm = torch.sigmoid(_mvt2_raw).permute(
                                2, 0, 1
                            ).contiguous()
                        else:
                            _flat = _mvt2_raw.permute(2, 0, 1).reshape(nc, h * w)
                            _mvt2_hm = torch.softmax(_flat, dim=-1).view(nc, h, w)
                        _anchor_valid = True
                        if (memory_inputs is not None
                                and "anchor_mask" in memory_inputs):
                            _anchor_valid = bool(
                                memory_inputs["anchor_mask"][0].item()
                            )
                        _mvt2_dir = os.path.join(arm_dir, "mvt2")
                        os.makedirs(_mvt2_dir, exist_ok=True)
                        _step_label = (
                            os.path.basename(visualize_save_dir.rstrip(os.sep))
                            or "step"
                        )
                        _save_mvt2_memory_grid(
                            anchor_img_one=out_arm["mvt2_anchor_ori_img"][0, :, 3:6],
                            current_img_one=out_arm["mvt2_ori_img"][0, :, 3:6],
                            current_hm_one=_mvt2_hm,
                            anchor_valid=_anchor_valid,
                            save_path=os.path.join(
                                _mvt2_dir, "anchor_memory_grid.png"
                            ),
                            view_names=_view_names,
                            step_label=f"(arm {arm_tag}  {_step_label})",
                        )
                except Exception as _e:
                    import traceback
                    print(f"[RMBench eval viz] arm {a_idx} failed: {_e}")
                    traceback.print_exc()

        _diag_dir = diag_save_dir or (visualize_save_dir if visualize else "")
        if visualize and self.memory_enabled and visualize_save_dir:
            try:
                _step_label = (
                    os.path.basename(visualize_save_dir.rstrip(os.sep))
                    or "step"
                )
                self._save_eval_memory_grid(
                    visualize_save_dir,
                    step_label=f"({_step_label})",
                    pc_world_for_bank=pc_world_for_bank,
                    img_feat_for_bank=img_feat_for_bank,
                )
            except Exception as _e:
                import traceback
                print(f"[RMBench eval viz] memory_grid failed: {_e}")
                traceback.print_exc()

        if _diag_dir and pred_grips_for_viz:
            try:
                write_rmbench_gripper_txt(
                    _diag_dir,
                    left_pred=pred_grips_for_viz[0],
                    right_pred=(
                        pred_grips_for_viz[1]
                        if len(pred_grips_for_viz) > 1
                        else pred_grips_for_viz[0]
                    ),
                )
            except Exception as _e:
                import traceback
                print(f"[RMBench eval viz] gripper.txt failed: {_e}")
                traceback.print_exc()
        if (_diag_dir and zoom_diag_entries and _st_sca is not None
                and _zoom_half_extent is not None):
            try:
                write_rmbench_zoom_pt_count_txt(
                    _diag_dir,
                    per_arm=zoom_diag_entries,
                    st_sca=_st_sca,
                    zoom_half_extent=_zoom_half_extent,
                    threshold=_STAGE2_ZOOM_MIN_PTS,
                    img_sizes_w=getattr(
                        self._net_mod, "renderer_img_sizes_w", [2.0, 2.0]),
                )
            except Exception as _e:
                import traceback
                print(f"[RMBench eval viz] zoom_pt_count.txt failed: {_e}")
                traceback.print_exc()

        # ---- Memory bank end-of-step push (shared stage-1 tokens). ----
        if self.memory_enabled:
            curr_tokens = out.get("mvt1_paligemma_tokens", None)
            if curr_tokens is not None:
                curr_tokens = curr_tokens.detach()
            # Keyframe-discriminator gate: admit this frame only if predicted to be a subtask boundary. The
            # gate reads POST-memory features while the STORED KV is the PRE-memory snapshot, for causal cleanliness.
            gate = True
            _thr = 0.5
            _trig_prob = None
            _logit_val = None
            mem_logit = out.get("mem_logit", None)
            if mem_logit is not None:
                _disc_cfg = (getattr(self._net_mod, "memory_cfg", {}) or {}).get(
                    "discriminator", {}) or {}
                _thr = float(_disc_cfg.get("threshold", 0.5))
                # RMBench eval is single-env (bs=1); mean collapses the (bs,) logit to a scalar decision.
                _trig_prob = float(torch.sigmoid(mem_logit).mean().item())
                _logit_val = float(mem_logit.float().mean().item())
                gate = bool(_trig_prob > _thr)
            _is_first = self.memory_bank.first_frame()
            _bank_before = self.memory_bank.num_history()
            if _is_first:
                self.memory_bank.set_anchor(
                    [_p.detach() for _p in pc_world_for_bank],
                    [_f.detach() for _f in img_feat_for_bank],
                    curr_tokens,
                )
            self.memory_bank.push(
                curr_tokens,
                gate=gate,
                pc_world=[_p.detach() for _p in pc_world_for_bank],
                img_feat=[_f.detach() for _f in img_feat_for_bank],
            )
            _bank_after = self.memory_bank.num_history()

            # ---- Per-step memory-trigger diagnostic txt (one per step dir) ----
            # Records this step's raw logit, sigmoid prob, threshold and ADMIT/SKIP plus bank size before/after.
            if _diag_dir:
                try:
                    write_rmbench_memory_trigger_txt(
                        _diag_dir,
                        is_first_frame=_is_first,
                        disc_enabled=(mem_logit is not None),
                        mem_logit=_logit_val,
                        trigger_prob=_trig_prob,
                        threshold=_thr,
                        gate=gate,
                        bank_before=_bank_before,
                        bank_after=_bank_after,
                    )
                except Exception as _e:
                    import traceback
                    print(f"[RMBench eval viz] memory_trigger.txt failed: {_e}")
                    traceback.print_exc()

        return result

    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        # Eval-time per-view fusion always uses uniform 1/N averaging (adaptive weighting is a deferred ablation).
        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q,
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        if self.rot_6d:
            # 6D regression: Gram-Schmidt -> R -> quat xyzw, matching the discrete head's (bs, 4) numpy contract.
            R_pred = aug_utils.rotation_6d_to_matrix(rot_q)  # (bs, 3, 3)
            pred_rot_quat = aug_utils.matrix_to_quaternion_xyzw_np(
                R_pred.detach().cpu().numpy()
            )
        else:
            pred_rot = torch.cat(
                (
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ].argmax(1, keepdim=True),
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ].argmax(1, keepdim=True),
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
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

    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        # GT heatmap, gated by `use_modified_focal_loss`: False (default) = Gaussian softmax-normalized over
        # (h*w) paired with soft-label CE; True = peak-1 CenterNet Gaussian paired with sigmoid focal loss.
        if self.use_modified_focal_loss:
            action_trans = generate_centernet_hm_from_pt(
                wpt_img.reshape(-1, 2),
                (h, w),
                radius=HM_VLA_MIN_RADIUS,
            )
        else:
            action_trans = mvt_utils.generate_hm_from_pt(
                wpt_img.reshape(-1, 2),
                (h, w),
                sigma=self.gt_hm_sigma,
                thres_sigma_times=3,
            )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans


    def reset(self):
        # Called by RolloutGenerator at each eval episode start (a no-op during training); clears the memory bank.
        self.memory_bank.reset()
        self._last_predicted_action = None
        # Eval overlay-viz state is per-episode; drop leftovers so a new episode never inherits a stale step list.
        self._eval_viz_episode_dir = None
        self._eval_viz_step_labels = []
        self._eval_viz_instruction = None

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
