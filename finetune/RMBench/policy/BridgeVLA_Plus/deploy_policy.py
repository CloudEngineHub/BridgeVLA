"""
BridgeVLA dual-arm policy for RMBench (Client-Server eval).

Two processes (see eval_double_env.sh):
  * SERVER (bridgevla / gembench conda env): holds the real BridgeVLA agent
    (PaliGemma + point-renderer). ``get_model`` runs here and returns a
    ``BridgeVLAModelServer`` exposing RPC methods ``get_action`` / ``reset_model``.
  * CLIENT (RMBench conda env): runs the SAPIEN sim. ``eval`` runs here with
    ``model`` = ModelClient RPC proxy; it packs the 4-camera observation,
    calls ``get_action`` on the server, and drives the robot with
    ``take_action(action_type='ee')``.

Action convention (verified against RMBench envs/robot/robot.py):
  * The model predicts each arm's gripper-fingertip TCP-CENTER pose
    [xyz, quat_xyzw] + grip 0/1.
  * ``take_action(action_type='ee')`` expects the EE-LINK pose, NOT the TCP
    center: plan_path -> _trans_from_gripper_to_endlink adds
    R @ [0.12 - gripper_bias, 0, 0], and aloha-agilex gripper_bias == 0.12 so
    that term is ZERO — the engine drives the EE link straight to whatever pose
    we send. The server (bridgevla_rmbench_model.get_action) therefore converts
    the predicted TCP center back to the EE-link origin (subtracts the 0.12 m
    gripper offset) before returning, so the executed TCP coincides with the
    predicted heatmap point instead of landing 0.12 m below it.
  * Quaternion: RMBench ee control wants wxyz; bridgevla outputs xyzw, so we
    convert xyzw -> wxyz here.
  * 16-D ee action = [Lxyz(3), Lquat_wxyz(4), Lgrip(1), Rxyz(3), Rquat_wxyz(4), Rgrip(1)].

NOTE: module import must stay light (the client imports this module too).
Heavy deps (torch / bridgevla) are imported lazily inside ``get_model`` /
the server wrapper, which only run on the server.
"""

import numpy as np

# RMBench camera-group names -> the bridgevla CAMERAS order (head/front/left/right).
# These 4 are the SAME cameras the dataset trains on: 2 STATIC world cameras
# (head @ (-0.032,-0.45,1.35) LargeView, front @ (0,-0.45,0.85) D435) + the 2
# per-arm WRIST cameras (left/right, LargeView), whose extrinsics move with the
# arms every step. encode_obs ships each camera's own per-step intrinsic_cv /
# extrinsic_cv, so the server unprojects into a correct fused world point cloud
# regardless of the wrist motion. No third_view / observer camera is used.
CAMERA_HDF5_NAMES = {
    "head": "head_camera",
    "front": "front_camera",
    "left": "left_camera",
    "right": "right_camera",
}
CAMERAS = ["head", "front", "left", "right"]


def encode_obs(observation):
    """Pack the RMBench runtime obs into a JSON/numpy-serializable dict the
    server can unproject. We send raw RGB + depth + intrinsics/extrinsics per
    camera (depth->world projection happens server-side, reusing the exact
    training-time util so train/eval stay identical).
    """
    obs_in = observation["observation"]
    packed = {"cameras": {}}
    for cam in CAMERAS:
        g = CAMERA_HDF5_NAMES[cam]
        cam_obs = obs_in[g]
        packed["cameras"][cam] = {
            "rgb": np.ascontiguousarray(cam_obs["rgb"]).astype(np.uint8),
            "depth": np.ascontiguousarray(cam_obs["depth"]).astype(np.float64),
            "intrinsic_cv": np.ascontiguousarray(cam_obs["intrinsic_cv"]).astype(np.float64),
            "extrinsic_cv": np.ascontiguousarray(cam_obs["extrinsic_cv"]).astype(np.float64),
        }
    return packed


def get_model(usr_args):
    """SERVER-side: build and return the BridgeVLA model wrapper."""
    # Lazy heavy imports — only the server (gembench env) executes this.
    from bridgevla_rmbench_model import BridgeVLAModelServer

    base_path = usr_args.get("ckpt_setting") or usr_args.get("model_folder")
    model_epoch = usr_args.get("model_epoch", "last")
    return BridgeVLAModelServer(base_path=base_path, model_epoch=model_epoch,
                                usr_args=usr_args)


def eval(TASK_ENV, model, observation):
    """CLIENT-side: one keyframe step. Predict dual-arm TCP poses, execute."""
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    payload = {"obs": obs, "instruction": instruction}

    # ---- Overlay-viz / step-diag hook (eval_config.yml). ----
    # eval_overlay_viz: server writes heatmap overlays + gripper/zoom txt. The
    # overlays come purely from the action-time network forward + point-cloud
    # renderer (top/front/right network views), so this NEVER touches the
    # third_view camera — fully decoupled from recording.
    # eval_video_log (via eval_step_diag): same step dir layout under viz/, but
    # only gripper.txt + zoom_pt_count.txt + action_pose.txt + plan_status.txt
    # (no heatmap forward overhead).
    # All best-effort — never let viz break the eval step.
    viz_on = bool(getattr(TASK_ENV, "eval_overlay_viz", False))
    step_diag = viz_on or bool(getattr(TASK_ENV, "eval_step_diag", False))
    step_dir = None
    if step_diag:
        try:
            import os
            ep = int(getattr(TASK_ENV, "test_num", 0))
            step = int(getattr(TASK_ENV, "take_action_cnt", 0))  # pre-take_action
            diag_root = (
                getattr(TASK_ENV, "eval_step_diag_dir", None)
                or getattr(TASK_ENV, "eval_overlay_viz_dir", None)
            )
            episode_dir = os.path.join(str(diag_root), f"episode{ep}")
            step_label = f"step_{step:03d}"
            step_dir = os.path.join(episode_dir, step_label)
            payload.update({
                "viz_episode_dir": episode_dir,
                "viz_step_label": step_label,
                "diag_step_dir": step_dir,
            })
            if viz_on:
                payload["viz"] = True
        except Exception as e:
            print(f"[RMBench eval viz] client payload setup failed: {e}")
            step_dir = None

    res = model.call(func_name="get_action", obs=payload)
    # res: {"left": [x,y,z, qx,qy,qz,qw, grip], "right": [...]} — EE-link pose
    # (TCP center already converted back to the EE-link origin server-side), xyzw.
    left = np.asarray(res["left"], dtype=np.float64)
    right = np.asarray(res["right"], dtype=np.float64)

    # NOTE: overlay viz intentionally does NOT save a third_view scene frame —
    # third_view rendering is gated solely by eval_video_log (recording). When
    # video is off the third_view camera isn't rendered at all (lossless speedup),
    # so there's no free frame to grab and we must not force a ~20s render here.

    def to_ee(arm):
        xyz = arm[0:3]
        quat_xyzw = arm[3:7]
        grip = arm[7]
        # xyzw -> wxyz for RMBench ee control.
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return np.concatenate([xyz, quat_wxyz, [grip]])

    action = np.concatenate([to_ee(left), to_ee(right)])  # 16-D

    if step_diag and step_dir is not None:
        try:
            import os
            from envs.utils.eval_step_diag import write_action_pose_txt

            os.makedirs(step_dir, exist_ok=True)
            left_ee = right_ee = left_tcp = right_tcp = None
            if getattr(TASK_ENV, "is_dual_arm", True):
                left_ee = TASK_ENV.robot.get_left_ee_pose()
                left_tcp = TASK_ENV.robot.get_left_tcp_pose()
                right_ee = TASK_ENV.robot.get_right_ee_pose()
                right_tcp = TASK_ENV.robot.get_right_tcp_pose()
            write_action_pose_txt(
                step_dir,
                left_server=left, right_server=right, action_16d=action,
                left_ee_now=left_ee, right_ee_now=right_ee,
                left_tcp_now=left_tcp, right_tcp_now=right_tcp,
            )
            TASK_ENV.eval_current_step_diag_dir = step_dir
        except Exception as e:
            print(f"[RMBench eval diag] action_pose.txt failed: {e}")

    # gripper_after_motion: first move to the predicted pose, and only then open/close the gripper
    # (rather than interpolating the gripper during the motion); both phases still count as one inference step.
    TASK_ENV.take_action(action, action_type="ee", gripper_after_motion=True)


def reset_model(model):
    """CLIENT-side hook. The client loop also calls model.call('reset_model')
    directly; this is kept for API completeness / local (non-C/S) eval."""
    try:
        model.call(func_name="reset_model")
    except AttributeError:
        pass
