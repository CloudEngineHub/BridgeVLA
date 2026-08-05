"""
SERVER-side BridgeVLA dual-arm model wrapper for RMBench eval.

Runs in the bridgevla (gembench) conda env. Holds the RVTAgent, exposes two
RPC methods to the sim client:
  * get_action(payload) -> {"left": [...8], "right": [...8]} TCP-center poses.
  * reset_model()       -> clears the episodic memory bank (per episode).

The depth->world point cloud projection reuses the SAME util as training
(RMBench_vla.utils.peract_utils_rmbench.depth_to_world_pcd) so the eval input
distribution matches training exactly.
"""
import glob
import os
import threading

import numpy as np
import torch
import yaml

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import bridgevla.config as default_exp_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as default_mvt_cfg
from bridgevla.mvt.mvt import MVT
from bridgevla.utils.rvt_utils import load_agent as load_agent_state

# RMBench_vla utils (CAMERAS / SCENE_BOUNDS / IMAGE_SIZE / projection). The
# train.sh-style PYTHONPATH puts finetune/RMBench_vla first; fall back to an
# explicit path insert so the server works even if only finetune/ is on path.
try:
    from utils.peract_utils_rmbench import (
        CAMERAS, SCENE_BOUNDS, IMAGE_SIZE, CAMERA_HDF5_NAMES, depth_to_world_pcd,
        world_to_render_pos, render_to_world_pos, tcp_center_to_endpose_pos,
    )
except Exception:  # pragma: no cover - path fallback
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "..", "..", "..", "RMBench_vla"))
    from utils.peract_utils_rmbench import (
        CAMERAS, SCENE_BOUNDS, IMAGE_SIZE, CAMERA_HDF5_NAMES, depth_to_world_pcd,
        world_to_render_pos, render_to_world_pos, tcp_center_to_endpose_pos,
    )

from PIL import Image

# Episode-end grid stitching of the per-step overlay tri-views. Lives in the
# bridgevla (server) env — the RMBench/sapien client env has no matplotlib —
# so the client triggers it via the finalize_episode_viz RPC.
from bridgevla.utils.viz_utils import stitch_episode_overlays


def _resize_rgb(img, target_hw):
    if img.shape[0] == target_hw and img.shape[1] == target_hw:
        return img
    return np.asarray(Image.fromarray(img).resize((target_hw, target_hw), Image.BILINEAR))


def _resize_pcd(pcd, target_hw):
    if pcd.shape[0] == target_hw and pcd.shape[1] == target_hw:
        return pcd.astype(np.float32, copy=False)
    t = torch.from_numpy(np.ascontiguousarray(pcd)).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(target_hw, target_hw), mode="nearest")
    return t.squeeze(0).permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def _read_eval_memory_switches():
    """Read the eval-side memory ablation switches from eval_config.yml.

    Source priority: the frozen snapshot exported by eval_double_env.sh
    (``RMBENCH_EVAL_CONFIG_SNAPSHOT``) > the eval_config.yml living next to this
    file. Returns ``(memory_temporal, memory_spatial)`` as bools (default True
    when the file / keys are absent — i.e. expect full memory).
    """
    cfg_path = os.environ.get("RMBENCH_EVAL_CONFIG_SNAPSHOT") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_config.yml"
    )
    cfg = {}
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:  # pragma: no cover - best effort, default full
            print(f"[BridgeVLA-RMBench] WARN: failed to read eval_config "
                  f"({cfg_path}): {e}; assuming full memory at eval.")
            cfg = {}
    t = cfg.get("memory_temporal", True)
    s = cfg.get("memory_spatial", True)
    return (True if t is None else bool(t)), (True if s is None else bool(s))


def _assert_eval_memory_matches_model(mvt_cfg):
    """Fail fast if the eval-side memory switches disagree with the loaded
    model's memory config (saved in its mvt_cfg.yaml). The model architecture
    is fixed at train time; this guards against evaluating the wrong ckpt for a
    memory ablation (e.g. eval expects temporal memory but the ckpt was trained
    without it).

    Effective model memory = memory.enabled AND <group switch>:
      * temporal_memory -> stage-1 coarse memory (temporal + stage-1 anchor)
      * spatial_memory  -> stage-2 fine spatial anchor
    """
    mem = getattr(mvt_cfg, "memory", None)
    enabled = bool(getattr(mem, "enabled", False)) if mem is not None else False
    model_temporal = enabled and bool(getattr(mem, "temporal_memory", True))
    model_spatial = enabled and bool(getattr(mem, "spatial_memory", True))

    eval_temporal, eval_spatial = _read_eval_memory_switches()

    mismatches = []
    if eval_temporal != model_temporal:
        mismatches.append(
            f"  - temporal_memory (memory 1, stage-1 temporal memory): "
            f"eval_config.memory_temporal={eval_temporal} vs "
            f"model={model_temporal} "
            f"(memory.enabled={enabled}, "
            f"temporal_memory={bool(getattr(mem, 'temporal_memory', True))})"
        )
    if eval_spatial != model_spatial:
        mismatches.append(
            f"  - spatial_memory (memory 2, stage-2 spatial memory): "
            f"eval_config.memory_spatial={eval_spatial} vs "
            f"model={model_spatial} "
            f"(memory.enabled={enabled}, "
            f"spatial_memory={bool(getattr(mem, 'spatial_memory', True))})"
        )
    if mismatches:
        raise ValueError(
            "[BridgeVLA-RMBench] eval memory switches do NOT match the loaded "
            "model's memory config:\n" + "\n".join(mismatches) + "\n"
            "Fix eval_config.yml (memory_temporal / memory_spatial) to match "
            "the ckpt, or evaluate the matching ckpt."
        )
    print(f"[BridgeVLA-RMBench] memory switches OK "
          f"(temporal={model_temporal}, spatial={model_spatial}).")


def load_dual_agent(model_path, exp_cfg_path=None, mvt_cfg_path=None, device=0):
    device = f"cuda:{device}"
    assert model_path is not None and os.path.exists(model_path), model_path
    model_folder = os.path.dirname(model_path)

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    exp_cfg.merge_from_file(exp_cfg_path or os.path.join(model_folder, "exp_cfg.yaml"))
    # Dual-arm REQUIRES place_with_mean=False (asserted by the agent). Unlike
    # the single-arm loader, we do NOT flip it on at eval.
    exp_cfg.freeze()

    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    mvt_cfg.merge_from_file(mvt_cfg_path or os.path.join(model_folder, "mvt_cfg.yaml"))
    mvt_cfg.freeze()

    # Guard: eval-side memory ablation switches must match the trained model.
    _assert_eval_memory_matches_model(mvt_cfg)

    rvt = MVT(renderer_device=device, **mvt_cfg)
    agent = bridgevla_agent.RVTAgent(
        network=rvt.to(device),
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir="eval_run",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 1000)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )
    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()
    print("[BridgeVLA-RMBench] Agent ready.")
    return agent


class BridgeVLAModelServer:
    def __init__(self, base_path, model_epoch="last", usr_args=None):
        usr_args = usr_args or {}
        device = int(usr_args.get("device", 0))
        # Resolve checkpoint: base_path may be a dir (use model_{epoch}.pth /
        # model_last.pth) or a direct .pth file.
        if base_path is None:
            raise ValueError("BridgeVLA policy needs ckpt_setting=<model dir or .pth>")
        if os.path.isdir(base_path):
            cand = os.path.join(base_path, f"model_{model_epoch}.pth")
            if not os.path.exists(cand):
                cand = os.path.join(base_path, "model_last.pth")
            if not os.path.exists(cand):
                # A released ckpt directory usually holds a single weight (with the epoch in its name) and
                # has neither model_{model_epoch}.pth nor model_last.pth — when it is unique, just use it.
                only = sorted(glob.glob(os.path.join(base_path, "model_*.pth")))
                if len(only) == 1:
                    cand = only[0]
                    print(f"[BridgeVLA] using the only checkpoint in {base_path}: "
                          f"{os.path.basename(cand)}")
            model_path = cand
        else:
            model_path = base_path
        self.image_size = IMAGE_SIZE
        self.device = f"cuda:{device}"
        self.agent = load_dual_agent(model_path, device=device)
        # Overlay-viz bookkeeping: the client passes a per-step save dir via
        # get_action; we remember the current episode dir + step labels so
        # finalize_episode_viz can stitch the per-step tri-views into grids.
        self._viz_episode_dir = None
        self._viz_step_labels = []
        self._viz_instruction = None

    def reset_model(self):
        """Clear the episodic memory bank at episode start (and the per-episode
        overlay-viz state — the client calls this once per episode start)."""
        self.agent.reset()
        self._viz_episode_dir = None
        self._viz_step_labels = []
        self._viz_instruction = None
        return {"ok": True}

    @torch.no_grad()
    def get_action(self, payload):
        """payload = {"obs": <encoded cameras>, "instruction": str,
                      [optional viz keys]}.

        Returns {"left": [x,y,z, qx,qy,qz,qw, grip], "right": [...]} —
        TCP-center poses (xyzw) + grip 0/1.

        Overlay viz: when ``payload["viz"]`` is true the client also passes
        ``viz_episode_dir`` (episode root) + ``viz_step_label`` (e.g.
        ``step_003``); we run act_rmbench with visualize=True writing this
        step's per-arm overlay tri-views under <episode_dir>/<step_label>/,
        and remember the labels so finalize_episode_viz can stitch them.
        """
        obs = payload["obs"]
        instruction = payload["instruction"]
        cams = obs["cameras"]

        viz_on = bool(payload.get("viz"))
        viz_episode_dir = payload.get("viz_episode_dir")
        viz_step_label = payload.get("viz_step_label")
        diag_step_dir = payload.get("diag_step_dir")

        observation = {"language_goal": [[[instruction]]]}
        for cam in CAMERAS:
            c = cams[cam]
            rgb = np.asarray(c["rgb"], dtype=np.uint8)
            depth = np.asarray(c["depth"], dtype=np.float64)
            K = np.asarray(c["intrinsic_cv"], dtype=np.float64)
            E = np.asarray(c["extrinsic_cv"], dtype=np.float64)

            rgb_r = _resize_rgb(rgb, self.image_size)            # (H,W,3) uint8
            pc = depth_to_world_pcd(depth, K, E)                 # (H,W,3) m
            # World -> render frame (+90 deg about Z), identical to training.
            pc = world_to_render_pos(pc)
            pc_r = _resize_pcd(pc, self.image_size)              # (H,W,3)

            # bridgevla _preprocess_inputs runs stack_on_channel, which expects
            # (B, T, C, H, W) and merges T into channel. With T=1 that yields
            # (B, C, H, W). Match the actioner: build (1, C, H, W) then add the
            # T axis -> (1, 1, C, H, W).
            rgb_t = torch.from_numpy(
                np.transpose(rgb_r, (2, 0, 1))[None]
            ).float().to(self.device).unsqueeze(0)
            pc_t = torch.from_numpy(
                np.transpose(pc_r, (2, 0, 1))[None]
            ).float().to(self.device).unsqueeze(0)
            observation[f"{cam}_rgb"] = rgb_t
            observation[f"{cam}_point_cloud"] = pc_t

        if viz_on and viz_episode_dir and viz_step_label:
            step_dir = os.path.join(viz_episode_dir, viz_step_label)
            self._viz_episode_dir = viz_episode_dir
            self._viz_instruction = instruction
            if viz_step_label not in self._viz_step_labels:
                self._viz_step_labels.append(viz_step_label)
            result = self.agent.act_rmbench(
                observation,
                visualize=True,
                visualize_save_dir=step_dir,
                diag_save_dir=step_dir,
            )
        elif diag_step_dir:
            result = self.agent.act_rmbench(
                observation, diag_save_dir=diag_step_dir)
        else:
            result = self.agent.act_rmbench(observation)
        out = {}
        for arm in ("left", "right"):
            wpt, quat_xyzw, grip = result[arm]
            # The agent predicts the gripper-fingertip TCP CENTER in the RENDER
            # frame; map it back to the world frame. The quaternion is already
            # in the world frame (orientation was never rotated).
            wpt_world = render_to_world_pos(np.asarray(wpt, dtype=np.float32))
            quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
            # take_action('ee') -> plan_path -> _trans_from_gripper_to_endlink
            # adds R @ [0.12 - gripper_bias, 0, 0]; aloha-agilex gripper_bias ==
            # 0.12 so that term is ZERO, i.e. the engine drives the EE LINK to
            # whatever pose we send (NOT the TCP). Training targets the TCP
            # center (endpose + 0.12), so we must undo that 0.12 here and hand
            # back the EE-link origin; otherwise the real TCP lands 0.12 m ahead
            # (below, along the approach axis) of the predicted heatmap point.
            ee_link_world = tcp_center_to_endpose_pos(wpt_world, quat_xyzw)
            out[arm] = np.concatenate([
                np.asarray(ee_link_world, dtype=np.float32),
                quat_xyzw,
                np.asarray([grip], dtype=np.float32),
            ]).astype(np.float32)
        return out

    def finalize_episode_viz(self, payload=None):
        """Kick off stitching of this episode's per-step overlay tri-views into
        per-(arm,stage) grids (rows=steps, cols=views). Called by the client
        after the step loop ends. No-op when overlay viz wasn't active.

        The stitch runs in a background daemon thread and this RPC returns
        immediately: long episodes (80+ steps) produce matplotlib grids that
        take minutes to render, which used to blow the client's 30s socket
        timeout, kill the connection, and crash the next episode's RPC.
        State is snapshotted here because reset_model() clears it at the next
        episode's start. stitch_episode_overlays uses the OO matplotlib API
        (no pyplot global state), so it is safe alongside the per-step pyplot
        viz running on the request-handler thread."""
        episode_dir = self._viz_episode_dir
        step_labels = list(self._viz_step_labels)
        instruction = self._viz_instruction
        if not episode_dir or not step_labels:
            return {"ok": True, "stitched": False}
        try:
            net = self.agent._net_mod
            n_views = int(net.num_img)
            stages = ("mvt1", "mvt2") if self.agent.stage_two else ("mvt1",)
            arms = ["L", "R"][:int(getattr(net, "num_arms", 2))]
            renderer = getattr(net, "renderer", None)
            if getattr(renderer, "oblique_views", False):
                view_names = ["oblique_a", "oblique_b", "oblique_c"]
            else:
                view_names = ["top", "front", "right"]
        except Exception as e:
            print(f"[RMBench eval viz] stitch setup failed: {e}")
            return {"ok": False, "stitched": False}

        def _stitch():
            try:
                stitch_episode_overlays(
                    episode_dir, step_labels, stages=stages,
                    n_views=n_views, view_names=view_names, arms=arms,
                    instruction=instruction,
                )
                print(f"[RMBench eval viz] stitched grids -> {episode_dir}")
            except Exception as e:
                import traceback
                print(f"[RMBench eval viz] stitch failed: {e}")
                traceback.print_exc()

        threading.Thread(
            target=_stitch, daemon=True,
            name=f"rmbench-viz-stitch-{os.path.basename(str(episode_dir))}",
        ).start()
        return {"ok": True, "stitched": "async"}
