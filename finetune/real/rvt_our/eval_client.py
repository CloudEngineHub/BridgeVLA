"""BridgeVLA++ real-robot evaluation client.

Captures camera images, sends them to the Flask inference server
(``eval_flask_app.py``, expected at ``SERVER_URL``) and drives the Dobot arm to
execute the predicted keyposes::

    python eval_client.py          # interactive
    python eval_client.py --fast   # pure-inference benchmark

One process == one episode. The server mints the log folder on /reset, next to
the checkpoint being evaluated, and allocates ``ep<NNN>`` from the episode
folders already on disk so the count survives a server restart::

    <ckpt dir>/eval/<model_stem>/
        summary_by_variation.txt          success rate per variation
        <EVAL_VARIATION>/
            summary.txt  episodes.csv
            ep<NNN>_<mmdd_HHMMSS>_<instruction_slug>_<result>/
                episode_meta.txt  latency.csv  latency_summary.txt
                step_0000/{meta.txt, views/, frames/}

Press **[s]** for success or **[f]** for fail to end an episode; Ctrl-C or an
error ends it as ``aborted``. The verdict is stamped onto the folder name and
the running success rate of the variation is rebuilt and printed before exit.
To print it again later::

    python eval_summary.py <ckpt dir>/eval/<model_stem>
    curl -s localhost:5000/summary?format=text
"""

import os
import re
import json
import time
import base64

import cv2
import numpy as np
import torch
import requests
import open3d as o3d
from pynput import keyboard
from transforms3d.euler import euler2mat
from scipy.spatial.transform import Rotation

from botarm import Server, DobotController, Point
from utils.real_camera_utils import Camera, get_cam_extrinsic
from utils.peract_utils import IMG_STRIDE
from utils.motion_frame_recorder import MotionFrameRecorder
import eval_summary


# Configuration  (edit these to match your setup)
SERVER_URL       = "http://127.0.0.1:5000/predict"
EPISODE_LENGTH   = 100
DEVICE           = "cuda:0"

# Task to evaluate, and the generalization axis this batch of episodes tests.
# EVAL_VARIATION is free text and names the log sub-folder verbatim, so every
# value gets its own episode numbering and its own success rate; only a name
# that would break the path (empty, containing a separator, '.' or '..') is
# rejected. Common axes: basic (reference condition), height, distractor,
# combination, category, lighting, background.
INSTRUCTION      = "put the watermelon in the upper drawer"
EVAL_VARIATION   = "basic"
eval_summary.validate_variation(EVAL_VARIATION)

# Safety clipping ranges (metres, base frame)
Y_RANGE = (-0.6, 0.3)
Z_RANGE = (-0.1, 0.5)

# Client-side post-inference correction for the drawer / shelf tasks, whose
# executed keypose tends to stop short along the gripper's forward axis: the
# predicted TCP is shifted this many metres along tool +z (the arm->gripper
# direction) before safety clipping, so the Y/Z limits still hold. 0.0 disables
# it; the model, the server and the predicted quaternion are never touched.
TCP_FORWARD_OFFSET_M = 0.0
# Shelf instructions occur with either preposition ("on/in the top shelf").
TCP_OFFSET_INSTRUCTION_PATTERNS = (
    re.compile(r"\bin the\b.*\bdrawer\b", re.IGNORECASE),
    re.compile(r"\b(?:on|in) the\b.*\bshelf\b", re.IGNORECASE),
)


def instruction_needs_tcp_offset(instruction: str) -> bool:
    return any(p.search(instruction) for p in TCP_OFFSET_INSTRUCTION_PATTERNS)


def apply_tcp_forward_offset(pos_m: np.ndarray, quat_xyzw: np.ndarray,
                             offset_m: float) -> np.ndarray:
    """Translate *pos_m* by *offset_m* along the gripper approach axis.

    *quat_xyzw* is the server's TCP orientation in scipy scalar-last order.
    Tool +z is the arm->gripper forward direction, so a positive offset pushes
    deeper.
    """
    forward = Rotation.from_quat(quat_xyzw).apply([0.0, 0.0, 1.0])
    return pos_m + offset_m * forward.astype(pos_m.dtype)


# ---- Dobot arm network config (single-arm) ----------------------------------
# The defaults below are PLACEHOLDERS: replace them with the address of your
# own controller and workstation, or override per run without editing the file,
#   ARM_IP=... LOCAL_IP=... python eval_client.py
ARM_IP     = os.environ.get("ARM_IP", "192.168.1.6")        # robot controller
ARM_PORT   = int(os.environ.get("ARM_PORT", 29999))         # Dobot dashboard
LOCAL_IP   = os.environ.get("LOCAL_IP", "192.168.1.100")    # this workstation
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", 12345))

# While the arm physically moves to the predicted keypose, a background thread
# saves camera frames at the camera's native rate into the server's per-episode
# folder, under ``step_<N>/frames/`` so they never mix with its ``views/``.
# Nothing is recorded while waiting for inference or for user input. At episode
# end the frames are stitched into ``episode_video.mp4`` at the top of the
# (renamed) episode folder; exec_frames_to_video.py does the same offline.
RECORD_EXEC_FRAMES = True
STITCH_EXEC_VIDEOS = True
EXEC_VIDEO_FPS     = 30

# Pure-inference benchmark: the step loop becomes capture -> inference ->
# execute, with no point-cloud preview, no per-step prompts and no frame
# recording (the grab thread and JPG writes cost camera bandwidth mid-episode).
# Latency reporting stays on, and so does the pre-capture buffer flush -- that
# one guarantees a post-motion observation, which is eval correctness rather
# than visualisation. End the episode with [s] / [f], honoured once the current
# step finishes, or Ctrl-C to abort. Also settable per run with --fast.
FAST_INFERENCE = False


# Coordinate transform helpers
def convert_pcd_to_base(cam_type: str, pcd: np.ndarray) -> np.ndarray:
    """Transform point cloud from camera frame to robot base frame."""
    transform = get_cam_extrinsic(cam_type, "left")
    h, w = pcd.shape[:2]
    pts = pcd.reshape(-1, 3)
    pts_h = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    pts_base = (transform @ pts_h.T).T[:, :3]
    return pts_base.reshape(h, w, 3)


# Visualisation
VIS_SCENE_BOUNDS = [-1.3, -1.5, -0.1, 0.4, 0.7, 0.6]


def vis_pcd_with_end_pred(pcd, rgb, end_pose, pred_pose):
    """Visualise point cloud with current arm pose and predicted target."""
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    x_min, y_min, z_min, x_max, y_max, z_max = VIS_SCENE_BOUNDS
    mask = (
        (pcd_flat[:, 0] >= x_min) & (pcd_flat[:, 0] <= x_max) &
        (pcd_flat[:, 1] >= y_min) & (pcd_flat[:, 1] <= y_max) &
        (pcd_flat[:, 2] >= z_min) & (pcd_flat[:, 2] <= z_max)
    )
    pcd_flat, rgb_flat = pcd_flat[mask], rgb_flat[mask]

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    # current arm pose
    end_pose = [float(x) for x in end_pose]
    T_end = np.eye(4)
    T_end[:3, :3] = euler2mat(*np.deg2rad(end_pose[3:]), axes="sxyz")
    T_end[:3, 3] = np.array(end_pose[:3]) * 0.001
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axis_end.transform(T_end)

    # predicted pose
    if isinstance(pred_pose, str):
        pred_pose = [float(x) for x in pred_pose.strip("{}").split(",")]
    else:
        pred_pose = [float(x) for x in pred_pose]
    T_pred = np.eye(4)
    T_pred[:3, :3] = euler2mat(*np.deg2rad(pred_pose[3:]), axes="sxyz")
    T_pred[:3, 3] = np.array(pred_pose[:3]) * 0.001
    axis_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axis_pred.transform(T_pred)

    o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_end, axis_pred])


# Serialisation / HTTP query
def _ndarray_to_base64(arr: np.ndarray) -> dict:
    return {
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "dtype": str(arr.dtype),
        "shape": arr.shape,
    }


def serialize_observation(language_goal, rgb: torch.Tensor, pcd: torch.Tensor) -> str:
    """Serialise observation tensors to a JSON string for the server."""
    return json.dumps({
        "language_goal": language_goal,
        "rgb":  _ndarray_to_base64(rgb.detach().cpu().numpy()),
        "pcd":  _ndarray_to_base64(pcd.detach().cpu().numpy()),
    })


def query_policy(json_str: str, url: str = SERVER_URL):
    """Send an observation to the inference server.

    Returns (pos, quat, gripper, model_latency), where *model_latency* is the
    server-measured pure model forward time in seconds (no network / IO).
    """
    try:
        resp = requests.post(
            url, data=json_str,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"Server returned {resp.status_code}")
            return None, None, None, None

        result = resp.json()
        print("Server result:", result)
        if "error" in result:
            print(f"Server error: {result['error']}")
            return None, None, None, None

        return (
            np.array(result["target_pos"],        dtype=np.float32),
            np.array(result["target_quat"],        dtype=np.float32),
            np.array(result["target_gripper"][0],  dtype=np.float32),
            result.get("inference_latency"),
        )
    except Exception as e:
        print(f"query_policy failed: {e}")
        return None, None, None, None


# User prompts
# Episode exit keys, accepted at every prompt: [s] = success + exit,
# [f] = fail + exit. The verdict is stamped onto the episode folder name by
# report_episode_result().
def prompt_before_action() -> str:
    """Returns 'retry', 'execute', 'success', 'fail' or 'abort'."""
    while True:
        print("\n>>> Pre-action: [r]etry | [n] execute | "
              "[s] success+exit | [f] fail+exit <<<")
        try:
            cmd = input().strip().lower()
        except KeyboardInterrupt:
            return "abort"
        mapping = {"r": "retry", "n": "execute", "s": "success", "f": "fail"}
        if cmd in mapping:
            return mapping[cmd]
        print(f"Invalid input {cmd!r}.")


def prompt_after_action(bot, server) -> str:
    """Returns 'continue', 'success', 'fail' or 'abort'.

    x/z open/close the gripper for calibration only (not a step).
    """
    decision = [None]

    def on_key_press(key):
        try:
            ch = key.char
        except AttributeError:
            if key == keyboard.Key.enter:
                decision[0] = "continue"
            return
        if ch == "x":
            print("Calibration: opening gripper")
            bot.claws_control(1, server.modbusRTU)
        elif ch == "z":
            print("Calibration: closing gripper")
            bot.claws_control(0, server.modbusRTU)
        elif ch == "s":
            decision[0] = "success"
        elif ch == "f":
            decision[0] = "fail"

    print("\n>>> Action done. Enter = next step | [s] success+exit | "
          "[f] fail+exit | [x] open claw | [z] close claw (calibration) <<<")
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    try:
        while decision[0] is None:
            time.sleep(0.05)
        return decision[0]
    except KeyboardInterrupt:
        return "abort"
    finally:
        listener.stop()


def start_fast_exit_listener():
    """Background [s]/[f] watcher for FAST_INFERENCE mode.

    Fast mode has no per-step prompt to type the exit keys at, so a global
    listener carries them instead: a press at any time ends the episode with
    that verdict once the current step finishes. Returns ``(listener,
    verdict)``; ``verdict[0]`` flips from None on the first press and never
    changes again.
    """
    verdict = [None]

    def on_key_press(key):
        try:
            ch = key.char
        except AttributeError:
            return
        if verdict[0] is not None:
            return
        if ch == "s":
            verdict[0] = "success"
            print("\n[fast] 's' pressed — episode ends SUCCESS after "
                  "this step.")
        elif ch == "f":
            verdict[0] = "fail"
            print("\n[fast] 'f' pressed — episode ends FAIL after "
                  "this step.")

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    return listener, verdict


def prompt_final_result() -> str:
    """Ask for the episode verdict when the step budget ran out."""
    while True:
        print("\n>>> Episode length reached: [s] success | [f] fail <<<")
        try:
            cmd = input().strip().lower()
        except KeyboardInterrupt:
            return "aborted"
        if cmd in ("s", "f"):
            return {"s": "success", "f": "fail"}[cmd]
        print(f"Invalid input {cmd!r}.")


def print_tally(stats: dict, variation: str = ""):
    """Print the running tally for *variation* (from /finish's ``tally`` field).

    ``aborted`` / ``unfinished`` episodes are listed but kept out of the rate:
    an abort is a do-over, and scoring it as a failure would deflate the number.
    """
    if not stats:
        return
    rate = stats.get("success_rate")
    rate_s = "n/a" if rate is None else f"{rate * 100:.1f}%"
    print(f"\n[tally] {f'[{variation}] ' if variation else ''}"
          f"{stats.get('scored', 0)} episodes scored: "
          f"{stats.get('success', 0)} success / {stats.get('fail', 0)} fail"
          f"  →  success rate {rate_s}")
    skipped = [f"{stats[k]} {k}" for k in ("aborted", "unfinished")
               if stats.get(k)]
    if skipped:
        print(f"[tally] not counted: {', '.join(skipped)}")
    for t in stats.get("per_task", []):
        t_rate = t.get("success_rate")
        t_rate_s = "  n/a" if t_rate is None else f"{t_rate * 100:5.1f}%"
        print(f"[tally]  {t_rate_s}  {t.get('success', 0)}/{t.get('scored', 0)}"
              f"  {t.get('instruction', '')}")


def _retally_locally(log_dir: str):
    """Rebuild the summary files ourselves when /finish never ran.

    *log_dir* is the variation folder holding this episode; its parent gets the
    cross-variation roll-up refreshed too, so a server outage cannot leave the
    campaign-level table stale.
    """
    try:
        stats = eval_summary.write_summary(log_dir)
        model_dir = os.path.dirname(log_dir)
        if eval_summary.is_model_dir(model_dir):
            eval_summary.write_rollup(model_dir)
        print_tally(stats, os.path.basename(log_dir))
    except Exception as e:
        print(f"WARNING: tally rebuild failed: {e}; run "
              f"`python eval_summary.py {log_dir}` later.")


def _resolve_final_dir(session_dir, result, server_reported=None):
    """Best-effort absolute path of the episode folder after the result stamp.

    Returns the path the server reported, else the conventional ``…_<result>``
    sibling, else the original (un-renamed) dir; None when none of those is on
    disk, in which case there is nothing local to stitch.
    """
    candidates = [server_reported]
    if session_dir:
        candidates.append(session_dir.rstrip(os.sep) + "_" + result)
        candidates.append(session_dir)
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return cand
    return None


def report_episode_result(session_dir, result: str):
    """Stamp the episode result onto the eval-run folder name and re-tally.

    POST /finish makes the server append *result* to ``episode_meta.txt``,
    rename the folder to ``…_<result>`` and rebuild the campaign tally next to
    the checkpoint; the returned tally is printed here. If the server is
    unreachable, the rename and the tally happen in this process instead
    (client and server share a filesystem in the default same-machine setup),
    so the episode still gets counted.

    Returns the absolute path of the episode folder after the rename, or None
    if it cannot be located, so the caller can stitch the recorded frames.
    """
    finish_url = SERVER_URL.rsplit("/", 1)[0] + "/finish"
    try:
        resp = requests.post(finish_url, json={"result": result}, timeout=10)
        resp.raise_for_status()
        payload = resp.json() or {}
        new_dir = payload.get("session_dir")
        print(f"[result] episode {result.upper()} → "
              f"{new_dir or '(no session dir)'}")
        print_tally(payload.get("tally"), payload.get("variation") or "")
        return _resolve_final_dir(session_dir, result, new_dir)
    except Exception as e:
        print(f"WARNING: /finish failed ({e}); trying local rename.")
    if session_dir and os.path.isdir(session_dir):
        new_dir = session_dir.rstrip(os.sep) + "_" + result
        try:
            with open(os.path.join(session_dir, "episode_meta.txt"), "a") as f:
                f.write(f"result: {result}\n"
                        f"end_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            os.rename(session_dir, new_dir)
            print(f"[result] episode {result.upper()} → {new_dir}")
            _retally_locally(os.path.dirname(new_dir))
            return new_dir
        except OSError as e:
            print(f"WARNING: local rename failed: {e}")
            return _resolve_final_dir(session_dir, result)
    else:
        print(f"[result] episode {result.upper()} — "
              f"no local session dir to rename.")
    return _resolve_final_dir(session_dir, result)


# Per-step latency logging  (console [latency] lines → disk)
def log_step_latency(session_dir, step, attempt, model_s, roundtrip_s):
    """Append one /predict query's timings to ``<session_dir>/latency.csv``.

    One row per query, retries included (``attempt`` restarts at 0 each step)::

        time,step,attempt,model_ms,roundtrip_ms

    ``model_ms`` is the server-measured pure forward time (empty when the
    server does not report it); ``roundtrip_ms`` is the client wall clock
    around the HTTP call, serialization and network included. Never raises —
    latency logging must not kill a step.
    """
    if not session_dir:
        return
    path = os.path.join(session_dir, "latency.csv")
    try:
        write_header = not os.path.exists(path)
        with open(path, "a") as f:
            if write_header:
                f.write("time,step,attempt,model_ms,roundtrip_ms\n")
            model_ms = "" if model_s is None else f"{model_s * 1000:.1f}"
            f.write(f"{time.strftime('%H:%M:%S')},{step},{attempt},"
                    f"{model_ms},{roundtrip_s * 1000:.1f}\n")
    except OSError as e:
        print(f"WARNING: latency csv write failed: {e}")


def write_latency_summary(episode_dir, model_latencies, roundtrip_latencies):
    """Aggregate latency.csv into ``<episode_dir>/latency_summary.txt``.

    Called once at episode end, after the folder got its final ``…_<result>``
    name. Records whether FAST_INFERENCE was on, since interactive-mode numbers
    are not comparable to benchmark ones. Never raises.
    """
    if not (episode_dir and os.path.isdir(episode_dir)):
        return
    if not (model_latencies or roundtrip_latencies):
        return

    def _stats(name, values_s):
        if not values_s:
            return f"{name}: n/a\n"
        ms = np.array(values_s) * 1000.0
        return (f"{name}: mean {ms.mean():.1f} ms | min {ms.min():.1f} ms | "
                f"max {ms.max():.1f} ms over {len(ms)} queries\n")

    try:
        with open(os.path.join(episode_dir, "latency_summary.txt"), "w") as f:
            f.write(f"fast_inference: {FAST_INFERENCE}\n")
            f.write(_stats("model_forward (server-side, pure forward)",
                           model_latencies))
            f.write(_stats("roundtrip (client, incl. serialization + network)",
                           roundtrip_latencies))
    except OSError as e:
        print(f"WARNING: latency summary write failed: {e}")


# Episode-video stitching  (motion frames → mp4)
def stitch_episode_videos(episode_dir):
    """Stitch this episode's recorded motion frames into ``episode_video.mp4``.

    Called once the episode folder has its final ``…_<result>`` name; the video
    lands at the top of it, next to ``episode_meta.txt``, at the camera-native
    EXEC_VIDEO_FPS (i.e. real time). Frame layout and codec fallbacks are
    reused from ``exec_frames_to_video.py``. A no-op when nothing was recorded,
    and never raises — a stitching failure must not lose the episode result.
    """
    if not (episode_dir and os.path.isdir(episode_dir)):
        return
    try:
        import exec_frames_to_video as efv
        steps = efv.collect_steps(episode_dir)
        if not steps:
            print(f"[exec_frames] no motion frames under {episode_dir}; "
                  f"no video written.")
            return

        real_path, n = efv.write_video(
            steps, os.path.join(episode_dir, "episode_video.mp4"),
            EXEC_VIDEO_FPS)
        print(f"[exec_frames] {real_path}: {len(steps)} steps, {n} frames, "
              f"{n / EXEC_VIDEO_FPS:.1f}s @ {EXEC_VIDEO_FPS} fps (real time)")
    except Exception as e:
        print(f"WARNING: episode video stitching failed: {e}")


# Observation preprocessing (camera → tensor)
def preprocess_observation(camera_info: dict, device: str = DEVICE) -> tuple:
    """
    Convert raw camera data into tensors for the server.

    Returns:
        observation_raw : dict with numpy arrays (for visualisation)
        rgb_tensor      : (1, 3, H, W)  float tensor
        pcd_tensor      : (1, 3, H, W)  float tensor
    """
    # Match Real_Dataset stride-4 downsample; keep BGR for server-side norm.
    pcd_base = convert_pcd_to_base("3rd", camera_info["3rd"]["pcd"])
    rgb_hw3 = camera_info["3rd"]["rgb"]
    if IMG_STRIDE > 1:
        rgb_hw3 = rgb_hw3[::IMG_STRIDE, ::IMG_STRIDE]
        pcd_base = pcd_base[::IMG_STRIDE, ::IMG_STRIDE]

    rgb_bgr = cv2.cvtColor(rgb_hw3, cv2.COLOR_RGB2BGR)

    obs_raw = {"pcd": pcd_base.copy(), "rgb": rgb_bgr.copy()}

    rgb_t = torch.from_numpy(
        np.transpose(rgb_bgr, [2, 0, 1])
    ).to(device).unsqueeze(0).float().contiguous()

    pcd_t = torch.from_numpy(
        np.transpose(pcd_base, [2, 0, 1])
    ).to(device).unsqueeze(0).float().contiguous()

    return obs_raw, rgb_t, pcd_t


# Gripper state reading
def read_gripper_state(bot, server) -> str:
    """Read the gripper open/close state (1=closed, 0=open)."""
    raw = bot.claws_read_command(server.modbusRTU, 258, 1)
    match = re.search(r"\{(.*?)\}", str(raw))
    return match.group(1)


# Main evaluation loop
def _eval():
    instructions = [[[INSTRUCTION]]]
    print(f"Language instruction for this episode: {INSTRUCTION!r}")

    # Fast mode benchmarks the raw inference→execute cadence, so anything that
    # burns time or camera/CPU outside that path is forced off for the episode.
    record_frames = RECORD_EXEC_FRAMES and not FAST_INFERENCE
    stitch_videos = STITCH_EXEC_VIDEOS and not FAST_INFERENCE
    if FAST_INFERENCE:
        print("[fast] FAST_INFERENCE on — no visualisation, no prompts, no "
              "frame recording; every prediction executes immediately. "
              "Press [s]/[f] any time to end the episode, Ctrl-C to abort.")

    # Decided once per episode from the instruction; every other task runs the
    # server prediction untouched (see TCP_FORWARD_OFFSET_M above).
    use_tcp_offset = (TCP_FORWARD_OFFSET_M != 0.0
                      and instruction_needs_tcp_offset(INSTRUCTION))
    if use_tcp_offset:
        print(f"[tcp_offset] drawer/shelf instruction — every executed TCP "
              f"will be shifted {TCP_FORWARD_OFFSET_M * 100:.1f} cm along "
              f"the gripper forward axis (tool +z).")

    cameras = Camera(camera_type="3rd")

    # One entry per /predict query, retries included: the server-measured pure
    # forward time and the client-side wall clock around the same query. Both
    # are appended to <session_dir>/latency.csv as they happen and aggregated
    # into latency_summary.txt at episode end.
    model_latencies = []
    roundtrip_latencies = []

    # ---- reset the inference server's episodic-memory bank ----
    # A fresh client run == a fresh episode, so the long-lived server's memory
    # bank is cleared once here; without it, this episode would reuse the
    # previous one's anchor / history tokens. A FAILED /reset is fatal even
    # though it is a no-op for checkpoints with memory disabled: running a
    # memory checkpoint on a stale bank silently corrupts the episode and its
    # recorded result. /reset also names the run folder from the instruction and
    # allocates this episode's number, returning ``session_dir`` (shared by the
    # server's view PNGs and this client's motion frames) and ``episode_index``.
    reset_url = SERVER_URL.rsplit("/", 1)[0] + "/reset"
    try:
        resp = requests.post(reset_url,
                             json={"instruction": INSTRUCTION,
                                   "variation": EVAL_VARIATION},
                             timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"/reset failed ({e}) — refusing to start the episode: the "
            f"server may carry stale episodic memory from a previous "
            f"episode. Check the server is up at {reset_url} and retry."
        ) from e
    payload = resp.json() or {}
    session_dir = payload.get("session_dir")
    episode_index = payload.get("episode_index")
    print("Server memory bank reset for new episode.")
    if session_dir:
        print(f"Eval log dir: {session_dir}")

    # Episode numbers restart in every variation, so the tag carries both.
    ep_tag = (f"EPISODE {episode_index}" if episode_index else "EPISODE ?") \
        + f" [{EVAL_VARIATION}]"
    print(f"\n{'#' * 60}\n### {ep_tag}  |  {INSTRUCTION}\n{'#' * 60}")

    # Set by the [s]/[f] exit keys; anything that ends the run without a
    # verdict is reported as "aborted".
    episode_result = None

    # Reuses the already-open camera, grabbing frames only between
    # recorder.start()/stop() so it never runs concurrently with capture().
    recorder = None
    if record_frames:
        if not session_dir:
            raise RuntimeError(
                "RECORD_EXEC_FRAMES=True but /reset did not return "
                "session_dir. Is eval_flask_app.py up to date / reachable?"
            )
        zed_3rd = cameras.cams[cameras.camera_types.index("3rd")]
        recorder = MotionFrameRecorder(zed_3rd, session_dir)
        # episode_meta.txt belongs to the server — it mints the run dir on
        # /reset and knows the checkpoint. Writing it here would truncate that.
        print(f"[exec_frames] recording motion frames to {session_dir}")

    # ---- initialise robot arm ----
    server = Server(ARM_IP, ARM_PORT, LOCAL_IP, LOCAL_PORT)
    server.sock.connect((server.ip, server.port))
    bot = DobotController(server.sock)
    server.init_com(bot)
    bot._initialize(server)

    # Fast mode has no per-step prompts, so the [s]/[f] exit keys move to a
    # background listener that is polled between steps.
    fast_listener, fast_verdict = None, [None]
    if FAST_INFERENCE:
        fast_listener, fast_verdict = start_fast_exit_listener()

    try:
        for step in range(EPISODE_LENGTH - 1):
            print(f"\n{'='*50}\n=== {ep_tag} | STEP {step} ===\n{'='*50}")
            print(f"Instruction: {INSTRUCTION!r}")

            # ── retry loop ─────────────────────────────────
            attempt = 0   # /predict queries burnt on this step, incl. retries
            while True:
                # capture (flush buffer with 10 grabs)
                for _ in range(10):
                    time.sleep(0.1)
                    camera_info = cameras.capture()
                    time.sleep(0.1)

                # ZedCam.capture() returns numpy *views* into sl.Mat buffers
                # that are freed when capture() returns.  Force-copy NOW
                # before the backing memory is recycled.
                camera_info["3rd"] = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in camera_info["3rd"].items()
                }

                obs_raw, rgb_t, pcd_t = preprocess_observation(camera_info)

                # read arm state (safe to do after camera data is copied)
                current_gripper = read_gripper_state(bot, server)
                print(f"Gripper state (1=closed, 0=open): {current_gripper}")
                end_pose = bot.get_pose()

                # query server
                t0 = time.time()
                json_str = serialize_observation(
                    instructions[0][0][0], rgb_t, pcd_t,
                )
                target_pos, target_quat, target_gripper, model_latency = \
                    query_policy(json_str)
                roundtrip_s = time.time() - t0
                print(f"Round-trip time (incl. serialization + network): "
                      f"{roundtrip_s:.3f}s")
                assert target_pos is not None, "Server returned None"
                roundtrip_latencies.append(roundtrip_s)
                log_step_latency(session_dir, step, attempt,
                                 model_latency, roundtrip_s)
                if model_latency is not None:
                    model_latencies.append(model_latency)
                    print(f"[latency] step {step}: model inference "
                          f"{model_latency * 1000:.1f} ms (server-side, pure "
                          f"forward) | mean "
                          f"{np.mean(model_latencies) * 1000:.1f} ms over "
                          f"{len(model_latencies)} queries")
                _gripper_state_str = "CLOSE" if int(target_gripper) == 1 else "OPEN"
                print(f"Predicted next gripper: {_gripper_state_str} "
                      f"(training={int(target_gripper)}) → robot cmd: "
                      f"{1 - int(target_gripper)} ({_gripper_state_str})")

                # quaternion sign convention
                if target_quat[0] < 0:
                    target_quat = -target_quat

                # task-specific forward offset (drawer / shelf only)
                if use_tcp_offset:
                    pos_before = target_pos.copy()
                    target_pos = apply_tcp_forward_offset(
                        target_pos, target_quat, TCP_FORWARD_OFFSET_M)
                    print(f"[tcp_offset] +{TCP_FORWARD_OFFSET_M * 100:.1f} cm "
                          f"along gripper axis: {np.round(pos_before, 4)} → "
                          f"{np.round(target_pos, 4)} (m, base frame)")

                # safety clipping
                target_pos[1] = np.clip(target_pos[1], *Y_RANGE)
                target_pos[2] = np.clip(target_pos[2], *Z_RANGE)

                # convert metres → millimetres
                target_pos_mm = target_pos * 1000.0

                # gripper command (training: 0=open,1=close → robot: 1=open,0=close)
                target_gripper_cmd = 1 - int(target_gripper)

                target_point = Point(target_pos_mm, target_quat, target_gripper_cmd)

                print(f"Pos(mm): {target_pos_mm}  Quat: {target_quat}  "
                      f"Gripper: {target_gripper} → cmd {target_gripper_cmd}")

                if FAST_INFERENCE:
                    if fast_verdict[0]:
                        episode_result = fast_verdict[0]
                        raise StopIteration
                    decision = "execute"
                else:
                    vis_pcd_with_end_pred(
                        obs_raw["pcd"], obs_raw["rgb"],
                        end_pose, target_point.position_quaternion_claw,
                    )
                    decision = prompt_before_action()
                if decision == "retry":
                    print("Retrying current step ...\n")
                    attempt += 1
                    continue
                elif decision == "execute":
                    print("Executing action.\n")
                    break
                else:  # success / fail / abort — end the episode here
                    if decision in ("success", "fail"):
                        episode_result = decision
                    raise StopIteration

            # ── execute action ─────────────────────────────
            # Frames are recorded ONLY inside this block (the robot's actual
            # motion). stop() must run even if the move raises, otherwise the
            # capture thread would collide with the next cameras.capture().
            if recorder is not None:
                recorder.start(step, meta_lines=[
                    f"step: {step}",
                    f"instruction: {INSTRUCTION}",
                    f"target_pos_mm: {target_pos_mm.tolist()}",
                    f"target_quat: {target_quat.tolist()}",
                    f"gripper_cmd (1=open,0=close): {target_gripper_cmd}",
                    # the server-side view meta logs the raw prediction; this
                    # line explains why target_pos_mm differs when it is on
                    f"tcp_forward_offset_m: "
                    f"{TCP_FORWARD_OFFSET_M if use_tcp_offset else 0.0}",
                ])
            try:
                pos_response = bot.point_control(target_point)
                print(f"pos_response: {pos_response}")
                if pos_response == "Success":
                    bot.claws_control(target_gripper_cmd, server.modbusRTU)
                else:
                    print("Position control failed — skipping gripper.")
            finally:
                if recorder is not None:
                    n_frames, rec_fps = recorder.stop()
                    print(f"[exec_frames] step {step}: {n_frames} frames "
                          f"(~{rec_fps:.1f} fps) → {recorder.last_step_dir}")

            # ── post-action ────────────────────────────────
            if FAST_INFERENCE:
                if fast_verdict[0]:
                    episode_result = fast_verdict[0]
                    print(f"Episode marked {episode_result.upper()} by user.")
                    break
                continue  # no prompt — straight into the next step
            post = prompt_after_action(bot, server)
            if post in ("success", "fail"):
                episode_result = post
                print(f"Episode marked {post.upper()} by user.")
                break
            elif post == "abort":
                print("Episode aborted (Ctrl-C).")
                break
        else:
            # EPISODE_LENGTH exhausted without an explicit verdict
            episode_result = prompt_final_result()

    except StopIteration:
        print(f"\nEpisode ended by user: "
              f"{(episode_result or 'aborted').upper()}.")
    finally:
        if model_latencies:
            lat_ms = np.array(model_latencies) * 1000.0
            print(f"\n[latency] episode summary — model inference over "
                  f"{len(lat_ms)} queries: mean {lat_ms.mean():.1f} ms | "
                  f"min {lat_ms.min():.1f} ms | max {lat_ms.max():.1f} ms")
        print("Shutting down ...")
        if fast_listener is not None:
            fast_listener.stop()
        if recorder is not None:
            recorder.close()   # join capture thread before closing the camera
        # Once the recorder has flushed, stamp the verdict onto the folder
        # name, aggregate the latency rows and stitch the motion frames — all
        # before hardware teardown, so a hanging camera / arm shutdown cannot
        # lose the result. None of the three raises.
        final_dir = report_episode_result(session_dir, episode_result or "aborted")
        write_latency_summary(final_dir or session_dir,
                              model_latencies, roundtrip_latencies)
        if stitch_videos and record_frames:
            stitch_episode_videos(final_dir)
        try:
            cameras.stop()
            del cameras
            print("Camera closed.")
        except Exception as e:
            print(f"Camera close error: {e}")
        bot.interrupt_close()
        server.sock.close()
        server.app.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="BridgeVLA++ real-robot evaluation client")
    parser.add_argument(
        "--fast", action="store_true",
        help="pure-inference benchmark: skip visualisation, per-step "
             "prompts and frame recording; predictions execute immediately "
             "(see the FAST_INFERENCE config block)")
    if parser.parse_args().fast:
        FAST_INFERENCE = True
    _eval()
