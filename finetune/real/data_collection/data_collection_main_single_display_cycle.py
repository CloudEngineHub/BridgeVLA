import socket  
import struct  
import threading  
import time  
from queue import Queue, Full, Empty  
import argparse  
import binascii  
import os  
import re
from datetime import datetime  
import cv2  
try:
    import pyrealsense2 as rs
except ImportError:  # the wrist camera is off by default, so the SDK is optional (only --use_wrist errors)
    rs = None
import numpy as np
import pyzed.sl as sl  
from pynput import keyboard  
import pickle  
from PIL import Image  
import shutil
import subprocess
import errno
import transforms3d  # make sure transforms3d is importable

from dataset_naming import (
    DEFAULT_DATA_ROOT,
    require_data_root,
    require_file,
    resolve_episode_save_path,
)

# Hand-eye calibration extrinsics (.npy). There is no sensible default — every machine's calibration differs,
# so point this at your own file (or pass --extrinsics_file). Unset, startup fails rather than silently using
# the wrong extrinsics:  export REAL_CAM_EXTRINSICS=/abs/path/to/your/extrinsic_matrix.npy
DEFAULT_EXTRINSICS_FILE = os.environ.get("REAL_CAM_EXTRINSICS", "")

# Robot arm address and camera serials. The IP default is a PLACEHOLDER —
# replace it or pass --ip. A ZED serial of 0 means "the first camera the SDK
# reports"; the RealSense serial is only needed with --use_wrist.
DEFAULT_ARM_IP           = os.environ.get("ARM_IP", "192.168.1.6")
DEFAULT_REALSENSE_SERIAL = os.environ.get("REALSENSE_SERIAL", "")
DEFAULT_ZED_SERIAL       = os.environ.get("ZED_3RD_SERIAL", "0")

import sys

_RVT_OUR_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "rvt_our"))
if _RVT_OUR_DIR not in sys.path:
    sys.path.insert(0, _RVT_OUR_DIR)
from botarm import TOOL_INDEX, DASHBOARD_PORT

# uint16 depth quantisation for 3rd_cam_depth/{i}.npy: units per metre. 4000 -> 0.25 mm resolution over a
# 16.38 m range (observed scenes top out around 7 m, so nothing clips). Must match convert_pcd_to_depth.py
# and real_dataset.py's deprojection.
DEPTH_SCALE = 4000.0
DEPTH_MAX_U16 = 65535

# Arm mode table, shared by the HUD and the logs (module-level so class methods can reference it regardless
# of how the file is invoked).
ROBOT_MODES = {
    1: "Initialization State (ROBOT_MODE_INIT)",
    2: "Brake Open (ROBOT_MODE_BRAKE_OPEN)",
    3: "Power off (ROBOT_MODE_POWEROFF)",
    4: "Disabled (ROBOT_MODE_DISABLED)",
    5: "Idle (ROBOT_MODE_ENABLE)",
    6: "Backdrive Mode (ROBOT_MODE_BACKDRIVE)",
    7: "Running (ROBOT_MODE_RUNNING)",
    8: "Single Move (ROBOT_MODE_SINGLE_MOVE)",
    9: "Error Status (ROBOT_MODE_ERROR)",
    10: "Paused (ROBOT_MODE_PAUSE)",
    11: "Collision Triggered (ROBOT_MODE_COLLISION)",
}

# Before collecting, point these two at your own absolute paths (both fail loudly with a hint when unset):
#   export REAL_COLLECT_DATA_ROOT=/abs/path/to/your/real_collect
#   export REAL_CAM_EXTRINSICS=/abs/path/to/your/extrinsic_matrix.npy
#
#   python data_collection_main_single_display_cycle.py --display
#
# Keys: a = save one step | b = undo (moved to a recoverable trash folder)
#       c = start a new episode (same instruction auto-increments _N)
#       Ctrl+C / ESC = quit
# Episodes are written to ${REAL_COLLECT_DATA_ROOT}/{instruction_slug}_{N}/.
class ZedCam:
    def __init__(self, serial_number, resolution=None):  
        self.zed = sl.Camera()  
        self.init_zed(serial_number)  

        if resolution:
            self.img_size = sl.Resolution()
            self.img_size.height = resolution[0]
            self.img_size.width = resolution[1]
        else:
            self.img_size = self.zed.get_camera_information().camera_configuration.resolution

        self.intrinsic = self._read_intrinsic()

    def _read_intrinsic(self):
        """Left-camera (fx, fy, cx, cy) for ``self.img_size``.

        We store depth instead of the full XYZ point cloud (the X/Y channels of
        MEASURE.XYZRGBA are exactly (u-cx)/fx*Z and (v-cy)/fy*Z, so they are
        pure redundancy — 24.9 MB/frame down to 4.1 MB). The trainer rebuilds
        the cloud by deprojecting with these numbers, so they must be saved
        alongside every episode.

        The SDK reports calibration for the *native* resolution; when
        retrieve_measure() is asked for a different size it rescales the image,
        so the intrinsics have to be rescaled by the same factor.
        """
        info = self.zed.get_camera_information()
        cam = info.camera_configuration.calibration_parameters.left_cam
        native = info.camera_configuration.resolution
        sx = float(self.img_size.width) / float(native.width)
        sy = float(self.img_size.height) / float(native.height)
        return (float(cam.fx) * sx, float(cam.fy) * sy,
                float(cam.cx) * sx, float(cam.cy) * sy)

    def verify_intrinsic(self, pcd_m, tol=1e-4):
        """Check that deprojecting Z with self.intrinsic reproduces X and Y.

        Guards against any convention mismatch between the SDK's reported
        calibration and the XYZRGBA buffer (cropping, rectification, a
        resolution rescale we got wrong). Called once per session — if it fails
        the depth-only files would be silently wrong, which is far worse than
        refusing to start.
        """
        fx, fy, cx, cy = self.intrinsic
        z = pcd_m[..., 2]
        valid = np.isfinite(z) & (z > 0.05)
        if valid.sum() < 1000:
            return None
        h, w = z.shape
        u = np.arange(w, dtype=np.float64)[None, :]
        v = np.arange(h, dtype=np.float64)[:, None]
        ex = np.abs(((u - cx) / fx) * z - pcd_m[..., 0])[valid].max()
        ey = np.abs(((v - cy) / fy) * z - pcd_m[..., 1])[valid].max()
        err = float(max(ex, ey))
        if err > tol:
            raise RuntimeError(
                f"ZED intrinsic check failed: deprojected XY differs from the "
                f"XYZRGBA buffer by {err:.3e} m (tol {tol:.1e}). Saved depth "
                f"would not reconstruct the point cloud — refusing to collect."
            )
        return err

    def init_zed(self, serial_number):
        init_params = sl.InitParameters()  
        init_params.set_from_serial_number(serial_number)  
        init_params.camera_resolution = sl.RESOLUTION.HD1080  
        init_params.camera_fps = 30  
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL  
        init_params.coordinate_units = sl.UNIT.MILLIMETER  

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            # exit() would skip the caller's cleanup; raise so __main__'s finally does the teardown.
            raise RuntimeError(f"ZED camera open failed: {err!r}")

        image = sl.Mat()  
        for _ in range(50):  
            runtime_parameters = sl.RuntimeParameters()  
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  
                self.zed.retrieve_image(image, sl.VIEW.LEFT)  

    def capture(self, max_attempts=15):
        image = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)
        depth_map = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)
        point_cloud = sl.Mat()

        # Bounded retry: a camera fault must not spin forever (the original `while True` held zed_lock and
        # deadlocked the save and display threads). On failure raise and let the caller drop the whole step.
        for _ in range(max_attempts):
            runtime_parameters = sl.RuntimeParameters()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, self.img_size)
                self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU, self.img_size)
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.img_size)
                frame_timestamp_ms = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_microseconds()
                break
        else:
            raise RuntimeError(f"ZED grab failed after {max_attempts} attempts")

        rgb_image = image.get_data()[..., :3]
        depth = depth_map.get_data()  
        depth[np.isnan(depth)] = 0  
        depth_image_meters = depth * 0.001  
        pcd = point_cloud.get_data()  
        pcd[np.isnan(pcd)] = 0  
        pcd = pcd[..., :3] * 0.001  

        return {  
            "rgb": rgb_image,  
            "depth": depth_image_meters,  
            "pcd": pcd,  
            "timestamp_ms": frame_timestamp_ms / 1000.0,  
        }  

    def stop(self):  
        self.zed.close()  

class CR5Realtime:
    # --- Dobot 30004/30005/30006 realtime feedback packet constants (official TCP-IP protocol) ---
    PACKET_SIZE = 1440                        # fixed 1440 bytes per packet
    TEST_VALUE_OFFSET = 48                    # offset of this packet's check value (48..55)
    FEEDBACK_TEST_VALUE = 0x0123456789ABCDEF  # the official check value (same as dobot_api.py)
    FEEDBACK_TEST_BYTES = struct.pack('<Q', FEEDBACK_TEST_VALUE)

    POSE_STALE_SEC = 0.5      # refuse to save when the pose snapshot is older than this (feed-interruption guard)
    KEY_DEBOUNCE_SEC = 0.3    # key debounce interval (shared by a/b/c and the teleop keys)
    DASHBOARD_TIMEOUT = 5.0   # response timeout for a single Dashboard command
    CLAW_POLL_INTERVAL = 0.5  # background gripper-state poll interval (saving reads it live as well)
    UNDO_TRASH_DIRNAME = "_undo_trash"  # 'b' moves undone data into this recoverable bin under data_root
    DISPLAY_MAX_ERRORS = 5    # consecutive display-thread errors before the display is turned off (collection is unaffected)

    # Every file written per time step (subfolder, extension). The post-save integrity check, the rollback on
    # failure, and the 'b' undo all read this one list, so the three can never disagree.
    STEP_FILES = (
        ("actions", "pkl"),
        ("3rd_cam_imgs", "png"),
        ("3rd_cam_rgb", "pkl"),
        ("3rd_cam_depth", "npy"),
    )

    # Wrist RealSense file list. Off by default — the downstream organize_dataset / training / visualization
    # only consume 3rd_cam_* + actions — and merged into self.STEP_FILES only with --use_wrist.
    WRIST_STEP_FILES = (
        ("wrist_cam_imgs", "png"),
        ("wrist_cam_rgb", "pkl"),
    )

    def __init__(self, ip=DEFAULT_ARM_IP, port=30004, frequency=None, data_root=None, base_save_path=None,
                 realsense_serial=DEFAULT_REALSENSE_SERIAL, zed_serial=int(DEFAULT_ZED_SERIAL),
                 instruction="put_bottle_in_microwave",
                 extrinsics_file=DEFAULT_EXTRINSICS_FILE, display=False,
                 use_wrist=False):
        # Take the single-instance lock first: two collector processes (including a leftover zombie) driving one
        # arm would trample the controller's global state (Tool/Modbus/ClearError).
        self._instance_lock_sock = None
        self._acquire_single_instance_lock(ip)
        self._warn_conflicting_processes()

        if not data_root:
            raise ValueError(
                "data_root is required; set DATA_ROOT near INSTRUCTION in __main__"
            )
        self.data_root = data_root
        os.makedirs(self.data_root, exist_ok=True)
        if base_save_path is not None:
            self.base_save_path = base_save_path
        else:
            self.base_save_path = resolve_episode_save_path(self.data_root, instruction)
        os.makedirs(self.base_save_path, exist_ok=True)

        self.save_counter = 0  
        self.ip = ip  
        self.port = port  
        self.running = False  
        self.data_queue = Queue(maxsize=100)  
        self.callbacks = []
        self.use_wrist = bool(use_wrist)
        if self.use_wrist:
            self.STEP_FILES = self.STEP_FILES + self.WRIST_STEP_FILES
        self.realsense_camera = self._init_realsense_camera(realsense_serial) if self.use_wrist else None
        self.zed_camera = ZedCam(serial_number=zed_serial)
        self.zed_lock = threading.Lock()  # guards ZED grabs so the display and save threads never race
        self.last_pose = None  
        self.dragging = False  
        self.claw_status = None  
        self.last_claw_status = None  
        self.save_lock = threading.Lock()
        self.latest_tcp_pose = None
        self.latest_pose_time = None  # monotonic time of the last successfully parsed pose (freshness check)
        self.save_worker_lock = threading.Lock()  # a/b/c are mutually exclusive: one at a time, no queueing, no keyboard stall
        self._key_debounce = {}  # key debounce timestamps
        self._debounce_note = {}  # throttle timestamps for the debounce notices (so holding a key doesn't spam)
        self.claw_thread = None  # background gripper-poll thread (separate from the pose-processing thread)
        self.last_event = ""  # most recent event, overlaid on the display HUD
        self._stopped = False  # stop() idempotency flag (both the ESC and Ctrl+C paths call it)
        self.key_listener = None
        self.instruction = instruction  
        # Every episode copies it into the episode directory (see _save_static_files); a wrong or missing copy
        # would ruin the geometric calibration of the whole batch, so its existence is checked here.
        self.extrinsics_file_path = require_file(
            extrinsics_file, "REAL_CAM_EXTRINSICS", "--extrinsics_file",
            "camera extrinsics file (.npy)",
        )
        self.display = display  # whether to show the live image
        self.display_queue = Queue(maxsize=1)  # queue carrying frames to the display thread
        self.display_thread = None  # display thread
        self.display_running = False  # display-thread run state

        # Dobot Dashboard long-lived connection (gripper state needs ModbusRTU, not a direct pymodbus 502 connection).
        self.dashboard_sock = None
        self.dashboard_lock = threading.Lock()
        self.modbus_rtu_id = None
        self.modbus_tcp_id = None  # ModbusCreate(502) channel, needed by the gripper write commands

        # Teleoperation (merged from arm_teleop.py, reusing the same Dashboard connection).
        self.latest_robot_mode = None  # latest arm mode, used to wait for idle after a motion
        self.teleop_lock = threading.Lock()  # serialise teleop actions so motion/gripper commands never overlap
        self.teleop_initial_joint = [224, 21, -87, -18, 88, 37]  # home joint angles (degrees)

        self.current_task_id = 0
        self.task_running = True

        if frequency is None:  
            if port == 30004:  
                self.frequency = 125  
            elif port == 30005:  
                self.frequency = 5  
            elif port == 30006:  
                self.frequency = 20  
            else:  
                self.frequency = 10  
        else:  
            self.frequency = frequency  

        self.period = 1.0 / self.frequency  
        self._stats = {
            'packets_received': 0,
            'packets_processed': 0,
            'invalid_packets': 0,   # packets that failed the checksum (misaligned/corrupt)
            'resyncs': 0,           # number of byte-stream re-alignments
            'dropped_bytes': 0,     # bytes discarded while re-aligning
            'last_latency': 0,
            'max_latency': 0,
            'start_time': 0
        }

        print(f"Initializing CR5 connection, IP: {ip}, Port: {port}, Frequency: {self.frequency}Hz")  
        self._connect()
        self._init_dashboard()
        self._set_tool_coordinate_index()
        self._init_modbus_rtu()

        self.folder_path = self.base_save_path
        self._init_episode_workspace()
        print(f"Episode save path: {self.folder_path}")

    def _init_episode_workspace(self):
        """Create actions/ and instruction files under the current episode folder."""
        os.makedirs(self.folder_path, exist_ok=True)
        self.actions_folder = os.path.join(self.folder_path, "actions")
        os.makedirs(self.actions_folder, exist_ok=True)

        self.instruction_file_path = os.path.join(self.folder_path, "instruction.txt")
        with open(self.instruction_file_path, 'w') as f:
            f.write(self.instruction)
        self.save_instruction_as_pkl()

        extrinsics_save_path = os.path.join(self.folder_path, "extrinsic_matrix.npy")
        shutil.copy(self.extrinsics_file_path, extrinsics_save_path)
        print(f"Extrinsics matrix copied to: {extrinsics_save_path}")

        self._save_intrinsic()

    def _save_intrinsic(self):
        """Write intrinsic.pkl for this episode (needed to rebuild the cloud).

        Since 3rd_cam_depth/{i}.npy stores only Z, an episode without these
        four numbers is unusable — so this runs at episode setup, before any
        frame is collected, and refuses to continue if the camera's reported
        calibration does not reproduce the XYZRGBA buffer.
        """
        if not getattr(self, "zed_camera", None):
            print("[intrinsic] ZED not initialized; skipping intrinsic.pkl")
            return
        fx, fy, cx, cy = self.zed_camera.intrinsic
        err = None
        try:
            # Same lock the save/display paths take — on the per-episode cycle
            # (_init_episode_workspace is re-entered for every new episode) the
            # display thread is already grabbing frames concurrently.
            with self.zed_lock:
                probe = self.zed_camera.capture()
            err = self.zed_camera.verify_intrinsic(probe["pcd"])
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[intrinsic] WARNING: could not verify against a live frame: {e}")

        meta = {
            "fx": np.array([fx], dtype=np.float64),
            "fy": np.array([fy], dtype=np.float64),
            "cx": np.array([cx], dtype=np.float64),
            "cy": np.array([cy], dtype=np.float64),
            "depth_scale": DEPTH_SCALE,
            "shape": (int(self.zed_camera.img_size.height),
                      int(self.zed_camera.img_size.width)),
            "num_frames": -1,          # constant K for the whole episode
            "source": "ZED SDK calibration_parameters.left_cam",
            "verify_err": err,
        }
        path = os.path.join(self.folder_path, "intrinsic.pkl")
        with open(path, "wb") as f:
            pickle.dump(meta, f)
        print(f"Intrinsics saved to: {path} "
              f"(fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}, "
              f"verify_err={err if err is None else format(err, '.2e')})")


    def save_instruction_as_pkl(self):
        instruction_data = self.instruction
        instruction_path = self.instruction_file_path.replace('.txt', '.pkl') 
        with open(instruction_path, 'wb') as pkl_file:
            pickle.dump(instruction_data, pkl_file)
        print(f"Converted instruction to PKL format and saved to {instruction_path}")

    def _init_realsense_camera(self, serial_number):
        """Initialise the wrist RealSense; called only with --use_wrist.

        Failures raise rather than being swallowed: if wrist images were explicitly requested, a missing
        camera should fail at startup. The old soft-failure returned None, the script started fine, and only
        on the first 'a' did _capture_realsense_frame raise "RealSense camera not initialized" — aborting
        every step so nothing could ever be saved.
        """
        if rs is None:
            raise RuntimeError("--use_wrist needs pyrealsense2, which is not installed in this environment")
        try:
            pipeline = rs.pipeline()
            config = rs.config()  
            config.enable_device(serial_number)  
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  
            profile = pipeline.start(config)  
            depth_sensor = profile.get_device().first_depth_sensor()  
            depth_scale = depth_sensor.get_depth_scale()  
            print(f"RealSense Depth Scale is: {depth_scale}")  
            return {  
                'pipeline': pipeline,  
                'profile': profile,  
                'depth_scale': depth_scale  
            }  
        except Exception as e:
            raise RuntimeError(
                f"failed to initialise the wrist RealSense (serial={serial_number}): {e}. "
                f"Check the device/serial, or drop --use_wrist to record ZED only."
            ) from e

    def _acquire_single_instance_lock(self, ip):
        """Take a process-level mutex keyed on the arm IP (a Linux abstract-domain socket, released on exit).

        Tool()/ModbusClose(0..3)/ClearError/EnableRobot on Dashboard (29999) are all controller-global state:
        two processes (even if one is a leftover zombie) connected at once overwrite each other's tool frame
        and close each other's Modbus channels — which shows up as a scrambled pose reference frame ("the TCP
        jumps to the origin") and mysterious gripper read/write failures. The kernel holds the lock, so it is
        released the moment the process dies and no stale lock file is left behind.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(f"\0bridgevla_dobot_collect_{ip}")
        except OSError as e:
            sock.close()
            if e.errno != errno.EADDRINUSE:
                # The lock mechanism itself is unavailable (rare): warn and continue rather than blocking collection.
                print(f"[WARN] single-instance lock unavailable ({e}); skipping the mutex check.")
                return
            raise SystemExit(
                "\n" + "=" * 68 + "\n"
                f"another collection process is already driving arm {ip}\n"
                "(or a leftover process from an unclean exit). Two processes on Dashboard(29999)\n"
                "overwrite each other's Tool/Modbus global settings, corrupting both pose and gripper data,\n"
                "so startup was refused.\n"
                "Diagnose:  pgrep -af data_collection_main_single_display_cycle\n"
                "Clean up:  pkill -f data_collection_main_single_display_cycle\n"
                "Restart this script once no leftover process remains.\n" + "=" * 68
            )
        self._instance_lock_sock = sock  # held until the process exits

    def _warn_conflicting_processes(self):
        """Warn about other scripts that might contend for Dashboard/Modbus (teleop, eval).

        They drive the same controller-global state. Teleop is already built into this script
        ('e'/'r'/'i'/'u'/'o'/'p'), so arm_teleop.py does not need to run alongside it.
        Warning only, never blocking (eval may be running on another machine against another arm).
        """
        try:
            result = subprocess.run(
                ["pgrep", "-af", "arm_teleop|eval_client|eval_flask_app"],
                capture_output=True, text=True, timeout=2,
            )
            procs = result.stdout.strip()
            if procs:
                print("=" * 68)
                print("[WARN] these processes may contend with the collector for the arm's Dashboard/Modbus:")
                print(procs)
                print("[WARN] gripper/home teleop is built in, so close the processes above before collecting.")
                print("=" * 68)
        except Exception:
            pass  # pgrep missing etc. must not block startup

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.settimeout(5.0)  
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  
        try:  
            self.sock.connect((self.ip, self.port))  
            print(f"Connected to CR5 robot arm: {self.ip}:{self.port}")  
            self._verify_data_stream()  
        except Exception as e:  
            print(f"Connection failed: {e}")  
            if self.port != 30004:  
                print(f"Try connecting to the standard real-time data port 30004")  
            raise  

    def _init_dashboard(self):
        """Open the long-lived Dobot Dashboard connection (29999), shared by Tool/ModbusRTU/GetHoldRegs."""
        self.dashboard_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dashboard_sock.settimeout(5.0)
        self.dashboard_sock.connect((self.ip, DASHBOARD_PORT))
        print(f"Connected to Dobot Dashboard: {self.ip}:{DASHBOARD_PORT}")

    def _drain_dashboard_socket(self):
        """Drain a stale response left in the buffer after a command timed out.

        Dashboard is strictly request/response, so once a read times out the leftover response makes every
        later command read the *previous* answer — which can decode the gripper state to the wrong value.
        """
        try:
            self.dashboard_sock.setblocking(False)
            stale = b''
            while True:
                try:
                    chunk = self.dashboard_sock.recv(4096)
                    if not chunk:
                        break
                    stale += chunk
                except (BlockingIOError, socket.error):
                    break
            if stale:
                print(f"[WARN] Dashboard dropped a stale response: {stale.decode(errors='replace').strip()!r}")
        finally:
            self.dashboard_sock.settimeout(self.DASHBOARD_TIMEOUT)

    def _dashboard_roundtrip(self, command: str) -> str:
        """Send one Dashboard command and read the response that actually belongs to it.

        Dobot replies as 'ErrorID,{value},CommandName(args);' — terminated by ';' and echoing the command
        name. A single recv(1024) could return half a response or the previous one, so we keep reading until
        the command echo plus a semicolon appear, discarding anything before the echo as leftover.
        """
        cmd_name = command.split('(', 1)[0].strip()
        self._drain_dashboard_socket()
        self.dashboard_sock.sendall(f"{command}\n".encode())

        buf = b''
        deadline = time.monotonic() + self.DASHBOARD_TIMEOUT
        self.dashboard_sock.settimeout(0.5)
        try:
            while time.monotonic() < deadline:
                try:
                    chunk = self.dashboard_sock.recv(1024)
                except socket.timeout:
                    continue
                if not chunk:
                    raise ConnectionError("Dashboard connection closed by robot")
                buf += chunk
                text = buf.decode(errors='replace')
                echo_pos = text.find(cmd_name)
                if echo_pos != -1:
                    semi_pos = text.find(';', echo_pos)
                    if semi_pos != -1:
                        start = text.rfind(';', 0, echo_pos) + 1
                        if start > 0:
                            print(f"[WARN] Dashboard dropped a crossed-over leftover: {text[:start]!r}")
                        return text[start:semi_pos + 1].strip()
        finally:
            self.dashboard_sock.settimeout(self.DASHBOARD_TIMEOUT)

        # Fallback: some firmware/commands don't echo the command name, so a complete semicolon-terminated response is accepted.
        text = buf.decode(errors='replace').strip()
        if ';' in text:
            resp = text[: text.rfind(';') + 1]
            print(f"[WARN] no command echo in the Dashboard response; truncating at the semicolon: {resp!r}")
            return resp
        raise TimeoutError(f"Dashboard response timeout for {command!r}: {buf!r}")

    def _send_dashboard_command(self, command: str) -> str:
        """Send a Dashboard command, reconnecting and retrying once on disconnect/timeout."""
        with self.dashboard_lock:
            try:
                return self._dashboard_roundtrip(command)
            except Exception as e:
                print(f"Dashboard command failed ({command}): {e}, reconnecting...")
                try:
                    self.dashboard_sock.close()
                except Exception:
                    pass
                self._init_dashboard()
                return self._dashboard_roundtrip(command)

    @staticmethod
    def _parse_dashboard_brace_value(response: str):
        match = re.search(r"\{([^}]+)\}", response)
        if not match:
            return None
        return match.group(1).strip()

    def _close_modbus_channels(self):
        """Close Modbus channels 0..3.

        If the previous session exited without closing them, the channels stay registered in the controller and
        the next ModbusCreate/ModbusRTUCreate returns -1 (already exists / resource busy). Clearing them first restores it.
        """
        for index in range(4):
            try:
                self._send_dashboard_command(f"ModbusClose({index})")
            except Exception:
                pass

    def _init_modbus_rtu(self):
        """Initialise the gripper Modbus channels (same order as arm_teleop.py).

        ModbusCreate(502) first, then ModbusRTUCreate: the TCP one carries the id-0 write inside the gripper
        open/close commands, and the RTU one reads the gripper state (GetHoldRegs 258).
        Stale channels are cleared first to avoid a -1.
        """
        self._close_modbus_channels()

        # Use self.ip rather than a hard-coded address, so --ip also redirects the gripper write commands.
        tcp_resp = self._send_dashboard_command(f'ModbusCreate("{self.ip}", 502, 2)')
        print(f"Init ModbusTCP: {tcp_resp}")
        tcp_val = self._parse_dashboard_brace_value(tcp_resp)
        if tcp_val is not None:
            self.modbus_tcp_id = int(tcp_val.split(",")[0].strip())
            print(f"ModbusTCP id = {self.modbus_tcp_id}")
        else:
            print("Warning: failed to parse ModbusTCP id; gripper teleop writes may fail.")

        resp = self._send_dashboard_command('ModbusRTUCreate(1, 115200, "N", 8, 1)')
        print(f"Init ModbusRTU: {resp}")
        val = self._parse_dashboard_brace_value(resp)
        if val is None:
            print("Warning: failed to parse ModbusRTU id; gripper reads will fail.")
            self.modbus_rtu_id = None
            return
        self.modbus_rtu_id = int(val.split(",")[0].strip())
        print(f"ModbusRTU id = {self.modbus_rtu_id}")

    def _ensure_robot_ready(self):
        """Clear alarms and re-enable the arm.

        If the previous session exited in drag-teach mode (Backdrive, mode 6) or an error state, Tool() and the
        motion commands return -1. ClearError() leaves drag mode / clears the error and EnableRobot() re-enables.
        """
        resp = self._send_dashboard_command("ClearError()")
        print(f"ClearError -> {resp}")
        resp = self._send_dashboard_command("EnableRobot()")
        print(f"EnableRobot -> {resp}")
        time.sleep(1.0)

    def _set_tool_coordinate_index(self):
        """Set global tool coordinate system via Dobot Dashboard (port 29999).

        Tool(TOOL_INDEX) decides whether the reported pose sits at the gripper TCP; on failure every collected
        label would be a flange pose, so this aborts rather than collecting bad data.
        """
        cmd = f"Tool({TOOL_INDEX})"
        response = self._send_dashboard_command(cmd)
        print(f"Dashboard: {cmd} -> {response}")
        if not response.startswith("0,"):
            # Usually the arm was left in drag mode (6) or an error state by the last session and Tool() returns
            # -1; clear errors, re-enable, and retry once.
            print("Tool() failed; retrying after ClearError() + EnableRobot() ...")
            self._ensure_robot_ready()
            response = self._send_dashboard_command(cmd)
            print(f"Dashboard (retry): {cmd} -> {response}")
        if not response.startswith("0,"):
            raise RuntimeError(
                f"Tool({TOOL_INDEX}) failed (response: {response}); "
                f"check DobotStudio tool coordinate calibration. Aborting data collection."
            )

    def _reconnect(self):
        """Re-establish the TCP connection to the arm after a disconnect/timeout."""
        self._notify("lost the arm's 30004 feedback stream, reconnecting ...", hud="30004 feed LOST, reconnecting...")
        while self.running:
            try:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.connect((self.ip, self.port))
                self._notify(f"reconnected to the arm's feedback stream: {self.ip}:{self.port}", hud="30004 feed reconnected")
                return
            except Exception as e:
                print(f"Reconnect failed: {e}, retrying in 2s...")
                time.sleep(2.0)

    def _verify_data_stream(self):
        try:
            self.sock.settimeout(2.0)
            # The first packet must be read in full: TCP is a byte stream, so a single recv(1440) can return half
            # a packet and leave the rest in the kernel buffer, permanently misaligning _recv_loop's 1440-byte
            # slicing — the pose at offset 624 then decodes to garbage (typically XYZ collapsing to ~0).
            data = b''
            while len(data) < self.PACKET_SIZE:
                chunk = self.sock.recv(self.PACKET_SIZE - len(data))
                if not chunk:
                    break
                data += chunk
            if len(data) == 0:
                print("Warning: Connected but no data received, check robot arm status")
            elif len(data) < self.PACKET_SIZE:
                print(f"Warning: incomplete first packet ({len(data)}/{self.PACKET_SIZE} bytes)")
            else:
                print(f"Successfully received data: {len(data)} bytes")
                if self._packet_is_valid(data):
                    print("Feedback packet check value OK (0x0123456789ABCDEF)")
                else:
                    print("Warning: feedback packet check value MISMATCH, will resync in recv loop")
                self._debug_data_format(data)
        except socket.timeout:
            print("Warning: No data stream received, robot arm may not be enabled or in error state")  
        except Exception as e:  
            print(f"Error verifying data stream: {e}")  
        finally:
            self.sock.settimeout(5.0)

    def _debug_data_format(self, data):  
        try:  
            msg_size = struct.unpack('<H', data[0:2])[0]  
            print(f"Message length of packet: {msg_size} bytes")  
            di_bytes = binascii.hexlify(data[8:16]).decode()  
            do_bytes = binascii.hexlify(data[16:24]).decode()  
            print(f"Digital input state (Hex): {di_bytes}")  
            print(f"Digital output state (Hex): {do_bytes}")  

            robot_mode = struct.unpack('<Q', data[24:32])[0]  
            print(f"Robot mode: {robot_mode}")  

            joint_positions = struct.unpack('<6d', data[432:480])  
            print("Joint positions (radians):")  
            for i, pos in enumerate(joint_positions):  
                print(f"  Joint {i + 1}: {pos:.4f} rad = {pos:.2f}°")  

            tcp_pose = struct.unpack('<6d', data[624:672])  
            print("TCP position (mm/degrees):")  
            print(f"  X: {tcp_pose[0]:.2f}, Y: {tcp_pose[1]:.2f}, Z: {tcp_pose[2]:.2f}")  
            print(f"  Rx: {tcp_pose[3]:.2f}, Ry: {tcp_pose[4]:.2f}, Rz: {tcp_pose[5]:.2f}")  

            print("Data packet format verification successful, can be parsed normally")  
        except Exception as e:  
            print(f"Failed to analyze packet format: {e}")  

    def pose_callback(self, data):  
        joints = data['joint_actual']  
        j_deg = [j * 57.2958 for j in joints]  

        cart = data['tcp_actual']  
        mode = data['robot_mode']  
        mode_str = ROBOT_MODES.get(mode, f"Unknown ({mode})")  

    def start(self):  
        self.running = True  
        self._stats['start_time'] = time.perf_counter()  

        self.recv_thread = threading.Thread(target=self._recv_loop, name="CR5-Receiver")  
        self.recv_thread.daemon = True  

        self.process_thread = threading.Thread(target=self._process_loop, name="CR5-Processor")  
        self.process_thread.daemon = True  

        self.recv_thread.start()
        self.process_thread.start()

        # Poll the gripper on its own thread, so a Dashboard stall never blocks the 30004 pose stream.
        if self.modbus_rtu_id is not None:
            self.claw_thread = threading.Thread(target=self._claw_poll_loop, name="CR5-ClawPoll")
            self.claw_thread.daemon = True
            self.claw_thread.start()

        print(f"Real-time data stream started, processing frequency: {self.frequency}Hz")

        if self.display:
            self.display_running = True
            self.display_thread = threading.Thread(target=self._display_loop, name="Display-Thread")
            self.display_thread.daemon = True
            self.display_thread.start()
            print("Live view display started")

    def _display_loop(self):
        """Display-thread main loop: live image + status HUD (step count / mode / feed / pose / last event)"""
        print("Display loop started")
        try:
            cv2.namedWindow('ZED Camera Live View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ZED Camera Live View', 960, 540)
        except Exception as e:
            print(f"[WARN] cannot create the display window (no graphical environment?): {e}; live display off, collection unaffected.")
            self.display_running = False
            return

        consecutive_errors = 0
        while self.display_running:
            try:
                # Fetch the newest image (mutually exclusive with the save thread's ZED access).
                with self.zed_lock:
                    result = self.zed_camera.capture()

                display_image = cv2.cvtColor(result["rgb"], cv2.COLOR_RGB2BGR)
                display_image = cv2.resize(display_image, (960, 540))
                self._draw_hud(display_image)
                cv2.imshow('ZED Camera Live View', display_image)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC: only set the flag; cleanup happens in the main thread's finally -> stop()
                    self._notify("display window received ESC, shutting down collection ...", hud="ESC pressed, shutting down")
                    self.running = False
                    self.display_running = False
                    break
                consecutive_errors = 0

            except Exception as e:
                # Occasional grab failures are tolerated; on repeated failure turn the display off rather than logging every 0.1 s.
                consecutive_errors += 1
                print(f"Error in display loop ({consecutive_errors}/{self.DISPLAY_MAX_ERRORS}): {e}")
                if consecutive_errors >= self.DISPLAY_MAX_ERRORS:
                    print("[WARN] repeated display-thread errors; live display turned off (collection and keys unaffected).")
                    self.display_running = False
                    break
                time.sleep(0.1)

        cv2.destroyAllWindows()
        print("Display loop stopped")

    def _draw_hud(self, img):
        """Overlay the collection status in the top-left of the preview (cv2.putText is ASCII-only)."""
        green, red, yellow = (80, 220, 80), (60, 60, 235), (60, 200, 235)

        age = None if self.latest_pose_time is None else time.monotonic() - self.latest_pose_time
        if age is None:
            feed_line, feed_color = "FEED: no data yet", red
        elif age > self.POSE_STALE_SEC:
            feed_line, feed_color = f"FEED: STALE {age:.1f}s (saves rejected)", red
        else:
            feed_line, feed_color = "FEED: OK", green

        pose = self.latest_tcp_pose
        if pose is None:
            pose_line, pose_color = "TCP: -", yellow
        else:
            ok, _ = self._validate_tcp_pose(pose)
            pose_line = f"TCP: X{pose[0]:7.1f} Y{pose[1]:7.1f} Z{pose[2]:7.1f} (mm)"
            pose_color = green if ok else red
            if not ok:
                pose_line += "  << BAD POSE"

        mode = self.latest_robot_mode
        mode_name = ROBOT_MODES.get(mode, f"Unknown({mode})") if mode is not None else "-"
        mode_short = mode_name.split(" (")[0]
        claw = self.claw_status
        if claw is None:
            claw_txt = "-"
        else:
            try:
                claw_txt = "OPEN" if int(claw) == 0 else "CLOSED"
            except (TypeError, ValueError):
                claw_txt = str(claw)

        lines = [
            (f"[{os.path.basename(self.folder_path)}]  steps saved: {self.save_counter}", green),
            (f"MODE: {mode_short}   CLAW: {claw_txt}", green if mode == 6 else yellow),
            (feed_line, feed_color),
            (pose_line, pose_color),
            ("a=save  b=undo  c=new-episode  e/r=claw  i/u=home  o=drag  p=clear-err  ESC=quit", yellow),
        ]
        if self.last_event:
            lines.append((self.last_event, yellow))

        y = 24
        for text, color in lines:
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 24

    def _packet_is_valid(self, packet: bytes) -> bool:
        """Validate one feedback packet per the official protocol: MessageSize==1440 and the check value at offset 48.

        Dobot's official demo (dobot_api.py) runs the same test_value check on every packet. Without it, blind
        1440-byte slicing means one misalignment (half packet / network glitch) turns the TCP pose at offset 624
        into pure garbage — typically XYZ≈0 while stationary, i.e. "the TCP collapses to the base origin".
        """
        if len(packet) < self.PACKET_SIZE:
            return False
        if struct.unpack('<H', packet[0:2])[0] != self.PACKET_SIZE:
            return False
        test_value = struct.unpack(
            '<Q', packet[self.TEST_VALUE_OFFSET:self.TEST_VALUE_OFFSET + 8]
        )[0]
        return test_value == self.FEEDBACK_TEST_VALUE

    def _resync_buffer(self, buffer: bytes) -> bytes:
        """Re-align a misaligned byte stream: find the check-value magic and set the header 48 bytes before it."""
        search_from = self.TEST_VALUE_OFFSET + 1  # skip the current (already invalid) alignment
        while True:
            idx = buffer.find(self.FEEDBACK_TEST_BYTES, search_from)
            if idx == -1:
                # No magic number in the buffer: keep only the tail (which may hold a partial magic/header).
                keep = self.TEST_VALUE_OFFSET + 8
                if len(buffer) > keep:
                    self._stats['dropped_bytes'] += len(buffer) - keep
                    return buffer[-keep:]
                return buffer
            start = idx - self.TEST_VALUE_OFFSET
            if start > 0:
                self._stats['resyncs'] += 1
                self._stats['dropped_bytes'] += start
                print(f"[WARN] feed misaligned; dropped {start} bytes to re-align "
                      f"(resync count: {self._stats['resyncs']})")
                return buffer[start:]
            search_from = idx + 1

    def _recv_loop(self):
        buffer = b''
        packet_size = self.PACKET_SIZE

        while self.running:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    print("Connection closed, attempting to reconnect...")
                    self._reconnect()
                    buffer = b''
                    continue

                buffer += chunk

                while len(buffer) >= packet_size:
                    packet = buffer[:packet_size]

                    # Official protocol checksum; a failure means the byte stream is misaligned, so re-sync.
                    if not self._packet_is_valid(packet):
                        self._stats['invalid_packets'] += 1
                        buffer = self._resync_buffer(buffer)
                        continue

                    buffer = buffer[packet_size:]

                    try:
                        self.data_queue.put((time.perf_counter(), packet), block=False)
                        self._stats['packets_received'] += 1
                    except Full:
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put((time.perf_counter(), packet), block=False)
                        except Exception:
                            pass

            except (socket.timeout, ConnectionResetError) as e:
                print(f"Connection error: {e}")
                self._reconnect()
                buffer = b''
            except Exception as e:
                # Other network errors (BrokenPipe/OSError) used to kill this thread silently, freezing
                # latest_tcp_pose so every later sample was an old pose. Reconnect instead.
                if not self.running:
                    break
                print(f"Receiver error: {e}")
                self._reconnect()
                buffer = b''

    def _process_loop(self):
        next_time = time.perf_counter() + self.period

        while self.running:
            current_time = time.perf_counter()

            if current_time >= next_time:
                # Drain the queue and keep only the newest packet: 30004 pushes at 125 Hz while we process at
                # self.frequency, so FIFO consumption would leave latest_tcp_pose up to queue_size/125 ≈ 0.8 s stale.
                latest = None
                while True:
                    try:
                        latest = self.data_queue.get(block=False)
                    except Empty:
                        break
                if latest is not None:
                    try:
                        timestamp, packet = latest
                        self._process_packet(timestamp, packet)
                        self._stats['packets_processed'] += 1
                    except Exception as e:
                        print(f"Error in process loop: {e}")
                next_time = current_time + self.period
            else:
                sleep_time = next_time - current_time
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.8)

    def _process_packet(self, timestamp, data):
        try:
            tcp_actual = struct.unpack('<6d', data[624:672])
            joint_actual = struct.unpack('<6d', data[432:480])
            robot_mode = struct.unpack('<Q', data[24:32])[0]
            self.latest_tcp_pose = tcp_actual
            self.latest_pose_time = time.monotonic()  # freshness timestamp, checked before saving
            self.latest_robot_mode = robot_mode
            is_dragging = robot_mode == 6  # 6 = Backdrive (drag teaching); 7 is Running (program motion)

            # The gripper state is polled by _claw_poll_loop. Inline polling used to stall pose handling for the
            # ~10 s a Dashboard timeout + reconnect takes, leaving latest_tcp_pose stale enough for the freshness
            # check to reject the whole step.

            if is_dragging != self.dragging:
                if is_dragging:
                    print("Detected operation, start recording...")
                    self.dragging = True
                else:
                    print("Operation stopped, stop recording...")
                    self.dragging = False

            for callback in self.callbacks:
                callback({
                    'joint_actual': joint_actual,
                    'tcp_actual': tcp_actual,
                    'robot_mode': robot_mode
                })

        except Exception as e:
            print(f"Error processing data: {e}")

    def manual_save(self):
        """Save data and images triggered by key press"""
        try:
            with self.save_lock:
                if not self.task_running:
                    print("Task is not running; this save was ignored.")
                    return
                if self.latest_tcp_pose is None:
                    self._notify("no pose received yet, save refused.", hud="Save REJECTED: no pose received yet")
                    return
                # Freshness check: while the receive thread is disconnected or reconnecting, latest_tcp_pose
                # stays at an old value, so saving would record a pose from before the keypress.
                pose_age = (
                    None if self.latest_pose_time is None
                    else time.monotonic() - self.latest_pose_time
                )
                if pose_age is None or pose_age > self.POSE_STALE_SEC:
                    age_str = "never updated" if pose_age is None else f"not updated for {pose_age:.1f}s"
                    hud_age = "never updated" if pose_age is None else f"{pose_age:.1f}s stale"
                    print("=" * 60)
                    print(f"Arm feedback stream interrupted or lagging (pose {age_str}); this save was refused.")
                    print("Check the network connection to the arm and press 'a' again once it recovers.")
                    print("=" * 60)
                    self._notify("save refused: pose is not fresh (see above).",
                                 hud=f"Save REJECTED: pose {hud_age}")
                    return
                self._save_data_and_image(self.latest_tcp_pose)
        finally:
            # Saving prints a lot; reprint the controls afterwards.
            self._print_controls_help()

    @staticmethod
    def _print_controls_help():
        """Print the key bindings. Reprinted after saving / heavy logging so it does not scroll away."""
        print("Press Ctrl+C (or ESC in the display window) to stop collecting")
        print("Press 'a' to save a data point")
        print("Press 'b' to undo the last data point (moved to the _undo_trash bin, recoverable)")
        print("Press 'c' to start a new episode (no new directory when the current one is empty)")
        print("--- Teleop (no need to run arm_teleop.py separately) ---")
        print("Press 'e' to close the gripper / 'r' to open it")
        print("Press 'i' to return home directly / 'u' to lift first (drag mode is exited automatically)")
        print("Press 'o' to enter drag teaching / 'p' to clear errors (leaves drag mode)")

    def _notify(self, msg, hud=None):
        """Print a timestamped event and mirror it onto the display HUD's last-event line.

        ``hud`` is the overlay text (cv2.putText can only draw ASCII); without it, ``msg`` is reused.
        """
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{stamp}] {msg}")
        self.last_event = f"[{stamp}] {hud if hud is not None else msg}"

    def _run_op(self, op_name, fn):
        """Run 'a'/'b'/'c' serially on a worker thread so the key-listener thread never blocks.

        Originally only 'a' used a worker: 'b'/'c' did disk I/O and waited on save_lock inside the pynput
        listener thread, so pressing 'b' mid-save froze the whole keyboard — including the 'p' emergency key.
        Now every op is refused with a message while another is running, rather than queueing or deadlocking.
        """
        if not self.save_worker_lock.acquire(blocking=False):
            self._notify(
                f"the previous operation has not finished, so '{op_name}' was ignored; try again shortly.",
                hud=f"Busy: previous op running, '{op_name}' ignored",
            )
            return

        def _run():
            try:
                fn()
            except Exception as e:
                self._notify(f"'{op_name}' failed: {e}", hud=f"Op '{op_name}' error: {e}")
            finally:
                self.save_worker_lock.release()

        threading.Thread(target=_run, name=f"Op-{op_name}", daemon=True).start()

    def _debounced(self, ch):
        """True when this key fired too soon after the last one (pynput repeats while held), so it should be ignored."""
        now = time.monotonic()
        last = self._key_debounce.get(ch, 0.0)
        self._key_debounce[ch] = now
        if (now - last) < self.KEY_DEBOUNCE_SEC:
            # Say explicitly that the key was ignored (rate-limited); silently dropping it looks like it worked.
            if now - self._debounce_note.get(ch, 0.0) > 1.0:
                self._debounce_note[ch] = now
                print(f"key '{ch}' repeated too fast / held down, ignored (debounce {self.KEY_DEBOUNCE_SEC:.1f}s).")
            return True
        return False

    def manual_undo(self):
        """'b' undoes one step: move the most recently written data into the trash bin and decrement the counter.
        'b' can be pressed repeatedly to walk back until the current task has no saved data left."""
        try:
            with self.save_lock:
                if not self.task_running:
                    self._notify("task is not running, cannot undo.", hud="Undo ignored: task not running")
                    return
                if self.save_counter <= 0:
                    self._notify("the current task has no data point to undo.", hud="Undo: nothing to undo")
                    return
                self._undo_last_step()
        finally:
            self._print_controls_help()

    def _step_file_paths(self, index):
        """Every file path that step `index` should have written (the list lives in STEP_FILES)."""
        return [
            os.path.join(self.folder_path, subdir, f"{index}.{ext}")
            for subdir, ext in self.STEP_FILES
        ]

    def _remove_step_files(self, index):
        """Delete every file written for step `index` and return how many were removed.
        Used only to roll back a failed save (half-step garbage is deleted outright); the 'b' key goes through
        _move_step_files_to_trash instead, so that data stays recoverable."""
        removed = 0
        for file_path in self._step_file_paths(index):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed += 1
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
        return removed

    def _undo_last_step(self):
        """Move every file of step (save_counter-1) into the trash bin and roll the counter back.

        Nothing is physically deleted: data undone by a mis-pressed 'b' can be restored verbatim (put the
        same-named files back into the episode's matching subdirectories). The trash directory name has no
        purely numeric suffix, so dataset scans (dataset_naming/organize_dataset) never mistake it for an episode.
        """
        index = self.save_counter - 1
        trash_dir = os.path.join(
            self.data_root,
            self.UNDO_TRASH_DIRNAME,
            os.path.basename(self.folder_path),
            f"step{index}_{datetime.now().strftime('%H%M%S')}",
        )
        moved, failed = self._move_step_files_to_trash(index, trash_dir)
        self.save_counter -= 1
        if failed:
            self._notify(
                f"undid step {index}, but {failed} file(s) could not be moved (leftovers are overwritten when that step is saved again)."
                f" Steps saved so far: {self.save_counter}",
                hud=f"Undo step {index} PARTIAL ({failed} failed), {self.save_counter} left",
            )
        else:
            self._notify(
                f"undid step {index} ({moved} file(s) moved to the trash bin, recoverable)."
                f" Steps saved so far: {self.save_counter}  Trash: {trash_dir}",
                hud=f"Undo step {index} -> trash, {self.save_counter} steps left",
            )

    def _move_step_files_to_trash(self, index, trash_dir):
        """Move every file of step `index` into trash_dir and return (moved, failed)."""
        moved = failed = 0
        for subdir, ext in self.STEP_FILES:
            src = os.path.join(self.folder_path, subdir, f"{index}.{ext}")
            if not os.path.exists(src):
                continue
            try:
                dst_dir = os.path.join(trash_dir, subdir)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, os.path.join(dst_dir, f"{index}.{ext}"))
                moved += 1
            except Exception as e:
                print(f"Failed to move {src} to trash: {e}")
                failed += 1
        return moved, failed

    def start_keyboard_listener(self):
        """Start keyboard listener thread.

        The listener thread only debounces and dispatches; anything slow (disk / camera / Dashboard) goes to the
        worker thread. Originally 'b'/'c' did file I/O and waited on save_lock right here, so pressing 'b'
        mid-save froze the whole keyboard, including the 'p' emergency key.
        """
        def on_press(key):
            try:
                ch = key.char
            except AttributeError:
                return  # Non-character key
            if ch is None:
                return
            if ch in ('a', 'b', 'c', 'e', 'r', 'i', 'u', 'o', 'p') and self._debounced(ch):
                return  # debounce held/repeated keys ('c' repeating would spawn a burst of new episodes)

            if ch == 'a':
                print("Key 'a' pressed! Saving data...")
                self._run_op('a', self.manual_save)
            elif ch == 'b':  # undo the last step -> move to the trash bin
                print("Key 'b' pressed! Undoing last data point...")
                self._run_op('b', self.manual_undo)
            elif ch == 'c':  # new episode
                print("Key 'c' pressed! Restarting task...")
                self._run_op('c', self.restart_task)
            # --- Teleop keys merged in from demon, reusing the same Dashboard connection ---
            elif ch == 'e':  # close the gripper
                print("Key 'e' pressed! Closing gripper...")
                self._trigger_teleop(self.teleop_close_claw)
            elif ch == 'r':  # open the gripper
                print("Key 'r' pressed! Opening gripper...")
                self._trigger_teleop(self.teleop_open_claw)
            elif ch == 'i':  # return home directly
                print("Key 'i' pressed! Returning to initial pose...")
                self._trigger_teleop(self.teleop_return_initial, lift=False)
            elif ch == 'u':  # lift first, then return home
                print("Key 'u' pressed! Lift then return to initial pose...")
                self._trigger_teleop(self.teleop_return_initial, lift=True)
            elif ch == 'o':  # enter drag teaching
                print("Key 'o' pressed! StartDrag...")
                self._trigger_teleop(self.teleop_start_drag)
            elif ch == 'p':  # clear errors / leave drag mode
                print("Key 'p' pressed! ClearError...")
                self._trigger_teleop(self.teleop_clear_error)

        self.key_listener = keyboard.Listener(on_press=on_press)
        self.key_listener.start()

    def _update_claw_status(self, new_status):  
        self.claw_status = new_status  

    def _read_claw_status(self) -> tuple:
        """Read gripper register 258 over Dashboard's ModbusRTU (matching botarm/eval).

        Register 258: 0 = fully open, non-zero (=1) = closed.
        Note: register 258 cannot be read with pymodbus straight over 502 — in drag mode the teach pendant's
        gripper actions are not reflected on the TCP Modbus channel, only on the ModbusRTU one.
        """
        if self.modbus_rtu_id is None:
            return False, "ModbusRTU not initialized."

        try:
            resp = self._send_dashboard_command(
                f'GetHoldRegs({self.modbus_rtu_id}, 258, 1, "U16")'
            )
            val = self._parse_dashboard_brace_value(resp)
            if val is None:
                return False, resp
            reg_val = int(val.split(",")[0].strip())
            return True, reg_val
        except Exception as err:
            print(f"Error reading claw status: {err}")
            return False, err

    def _claw_poll_loop(self):
        """Background gripper-state poll thread (saving also reads it live as a fallback).

        This used to be inlined in _process_packet, so a Dashboard timeout + reconnect (up to ~10s) stalled the
        30004 pose thread, leaving latest_tcp_pose stale enough that every 'a' was rejected by the freshness
        check. On its own thread the pose stream and the gripper read are fully decoupled.
        """
        consecutive_failures = 0
        while self.running:
            flag, status = self._read_claw_status()
            if flag:
                consecutive_failures = 0
                self._update_claw_status(status)
                interval = self.CLAW_POLL_INTERVAL
            else:
                consecutive_failures += 1
                # Back off on failure so an unavailable Dashboard doesn't spam logs or worsen congestion.
                interval = min(3.0, self.CLAW_POLL_INTERVAL * (1 + consecutive_failures))
            deadline = time.monotonic() + interval
            while self.running and time.monotonic() < deadline:
                time.sleep(0.1)

    # Teleoperation (merged from arm_teleop.py; all of it reuses the same Dashboard connection).
    def _claws_set_hold(self, mid, addr, count, value):
        """Equivalent to the reference claws_send_command: SetHoldRegs(mid, addr, count, {value}, \"U16\")."""
        self._send_dashboard_command(
            f'SetHoldRegs({mid}, {addr}, {count}, {{{value}}}, "U16")'
        )

    def teleop_open_claw(self):
        """Open the gripper (register 258 = 0)."""
        if self.modbus_rtu_id is None:
            print("Teleop: ModbusRTU not initialised, cannot drive the gripper.")
            return
        rtu = self.modbus_rtu_id
        tcp = self.modbus_tcp_id if self.modbus_tcp_id is not None else 0
        self._claws_set_hold(rtu, 258, 1, 0)
        self._claws_set_hold(rtu, 259, 1, 1)
        self._claws_set_hold(rtu, 264, 1, 1)
        self._claws_set_hold(tcp, 258, 1, 0)
        print("Teleop: gripper opened")

    def teleop_close_claw(self):
        """Close the gripper (register 258 = 1)."""
        if self.modbus_rtu_id is None:
            print("Teleop: ModbusRTU not initialised, cannot drive the gripper.")
            return
        rtu = self.modbus_rtu_id
        tcp = self.modbus_tcp_id if self.modbus_tcp_id is not None else 0
        self._claws_set_hold(rtu, 258, 1, 1)
        self._claws_set_hold(rtu, 259, 1, 0)
        self._claws_set_hold(rtu, 264, 1, 1)
        self._claws_set_hold(tcp, 258, 1, 1)
        print("Teleop: gripper closed")

    def _wait_until_idle(self, timeout=30.0):
        """Wait for the arm to finish moving and return to idle (5).

        The original version treated drag mode (6) as \"idle\" too: with MovJ rejected by the controller in drag
        mode this returned True immediately and the caller printed \"returned home\" — a false success. Drag mode
        is now exited automatically before homing; if the pendant switches back to drag mid-motion, the motion
        has ended or been aborted anyway, so True is returned with an explicit notice.
        """
        time.sleep(0.3)  # give the motion command a moment to reach Running (7)
        start = time.time()
        while time.time() - start < timeout:
            mode = self.latest_robot_mode
            if mode == 5:
                return True
            if mode == 6:
                print("Teleop: the arm entered drag mode while waiting; the motion has ended or been aborted.")
                return True
            time.sleep(0.1)
        print("Teleop: timed out waiting for the arm to go idle.")
        return False

    def _exit_drag_if_needed(self, timeout=3.0):
        """Leave drag teaching (Backdrive, mode 6) automatically; True means we are no longer in drag mode.

        The controller rejects MovJ outright while in drag mode (which is exactly why 'i' appeared to do nothing
        during collection), so drag must be exited first. StopDrag() is tried first, and ClearError() is the
        fallback if the mode still does not leave 6."""
        if self.latest_robot_mode != 6:
            return True
        self._notify("drag teaching detected, exiting it before homing ...", hud="Drag mode: auto-exiting before move")
        for cmd in ("StopDrag()", "ClearError()"):
            resp = self._send_dashboard_command(cmd)
            print(f"Teleop: {cmd} -> {resp}")
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.latest_robot_mode != 6:
                    self._notify("left drag teaching, homing now.", hud="Drag exited, moving to home")
                    return True
                time.sleep(0.1)
        self._notify("could not leave drag mode (the arm is still in Backdrive); homing cancelled.",
                     hud="Exit drag FAILED, move cancelled")
        return False

    def _teleop_movj(self, movj_args: str) -> bool:
        """Issue one MovJ and check ErrorID, reporting clearly when the controller rejects it."""
        resp = self._send_dashboard_command(f"MovJ({movj_args})")
        if not resp.startswith("0,"):
            self._notify(f"MovJ rejected by the controller: {resp}", hud=f"MovJ REJECTED: {resp}")
            return False
        return True

    def _get_pose_via_dashboard(self):
        """Read the current TCP pose (mm/deg) via Dashboard GetPose(); returns None when parsing fails."""
        resp = self._send_dashboard_command("GetPose()")
        val = self._parse_dashboard_brace_value(resp)
        if val is None:
            print(f"Teleop: could not parse GetPose: {resp}")
            return None
        try:
            return [float(x) for x in val.split(",")[:6]]
        except Exception as e:
            print(f"Teleop: GetPose raised: {e} ({resp})")
            return None

    def teleop_return_initial(self, lift=False):
        """Return to the home joint position; with lift=True, lift first and then translate.

        During collection the arm is usually in drag teaching (Backdrive), where the controller rejects MovJ,
        so drag mode is exited first. Press 'o' to re-enter drag teaching after homing.
        """
        if not self._exit_drag_if_needed():
            return
        if lift:
            pose = self._get_pose_via_dashboard()
            if pose is not None:
                x, y, z, rx, ry, rz = pose
                lift_pose = f"{{{x + 20:.4f},{y:.4f},{z + 150:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                if not self._teleop_movj(f"pose={lift_pose},a=30,v=30"):
                    return
                self._wait_until_idle()
                pose2 = self._get_pose_via_dashboard()
                if pose2 is not None:
                    x, y, z, rx, ry, rz = pose2
                    lift_pose2 = f"{{{x + 100:.4f},{y - 110:.4f},{z:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                    if not self._teleop_movj(f"pose={lift_pose2},a=30,v=30"):
                        return
                    self._wait_until_idle()
                print("Teleop: lifted, continuing home")

        joint_str = "{" + ",".join(f"{v:.4f}" for v in self.teleop_initial_joint) + "}"
        if not self._teleop_movj(f"joint={joint_str},a=30,v=30"):
            return
        self._wait_until_idle()
        print("Teleop: returned to the home joint position (press 'o' to resume drag collection)")

    def teleop_start_drag(self):
        resp = self._send_dashboard_command("StartDrag()")
        print(f"Teleop: StartDrag -> {resp}")

    def teleop_clear_error(self):
        resp = self._send_dashboard_command("ClearError()")
        print(f"Teleop: ClearError -> {resp}")

    def _trigger_teleop(self, action, *args, **kwargs):
        """Run one teleop action serially on its own thread, so the key listener never blocks and commands never overlap."""
        if not self.teleop_lock.acquire(blocking=False):
            print("Teleop: the previous teleop action has not finished; this key was ignored.")
            return

        def _run():
            try:
                action(*args, **kwargs)
            except Exception as e:
                print(f"Teleop failed: {e}")
            finally:
                self.teleop_lock.release()

        threading.Thread(target=_run, daemon=True).start()

    def restart_task(self):
        """End the current episode and allocate the next {slug}_{N} directory from the language instruction.

        No new directory is opened when the current episode has no saved step: otherwise a mis-pressed or held
        'c' would spray empty episode numbers across the dataset and pollute the organize_dataset / training scans.
        """
        with self.save_lock:
            if self.save_counter == 0:
                self._notify(
                    f"the current episode has not saved any data, so the directory is reused: {self.folder_path}",
                    hud="New episode skipped: current episode is empty",
                )
                self._print_controls_help()
                return
            self.task_running = False
            self.current_task_id += 1
            self.base_save_path = resolve_episode_save_path(self.data_root, self.instruction)
            self.folder_path = self.base_save_path
            self._init_episode_workspace()
            self.save_counter = 0
            self.task_running = True

        self._notify(
            f"new episode started (Task ID: {self.current_task_id}), data directory: {self.folder_path}",
            hud=f"New episode: {os.path.basename(self.folder_path)}",
        )
        self._print_controls_help()

    # Below this height (mm) the TCP frame transform is treated as broken. In normal operation the TCP height
    # is far above this; a network glitch between the collector and the arm makes the returned pose transform
    # incorrectly in the base frame, typically collapsing Z to ~0 and flattening the trajectory.
    TCP_Z_MIN_MM = 30.0

    def _validate_tcp_pose(self, tcp_actual):
        """Check that the TCP pose read from Dashboard was correctly transformed into the base frame.

        Returns (is_valid, reason). Known failure: when the network between collector and arm glitches, the
        returned TCP pose transforms incorrectly, typically collapsing Z (height, in mm) to ~0.
        """
        try:
            arr = np.asarray(tcp_actual[:6], dtype=float)
        except Exception:
            return False, f"pose could not be parsed as numbers: {tcp_actual!r}"
        if arr.shape[0] < 6 or not np.all(np.isfinite(arr)):
            return False, f"pose contains invalid values (NaN/Inf) or is too short: {tcp_actual!r}"
        x, y, z = arr[0], arr[1], arr[2]
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False, "pose (X,Y,Z) is all zero; no valid data was read"
        if abs(z) < self.TCP_Z_MIN_MM:
            return False, (
                f"abnormal Z height (Z={z:.2f}mm < {self.TCP_Z_MIN_MM:.0f}mm); "
                f"the TCP appears flattened onto the base plane (X={x:.2f}, Y={y:.2f})"
            )
        return True, ""

    def _save_data_and_image(self, tcp_actual):  
        print("Saving data")
        try:  
            position_x, position_y, position_z, roll, pitch, yaw = tcp_actual  
            print(position_x, position_y, position_z, roll, pitch, yaw)

            # --- TCP frame-transform sanity check (guards against network glitches) ---
            ok, reason = self._validate_tcp_pose(tcp_actual)
            if not ok:
                print("=" * 60)
                print("TCP frame transform is wrong; contact the algorithm team")
                print(f"Reason: {reason}")
                print("This data point was discarded and not written to disk (save_counter not incremented).")
                print("=" * 60)
                self._notify("save refused: TCP pose validation failed (reason above).",
                             hud="Save REJECTED: invalid TCP pose")
                return

            # Read the gripper live at save time rather than using the background cache.
            claw_flag, claw_raw = self._read_claw_status()
            if claw_flag:
                self.claw_status = claw_raw
                claw_status = 0 if int(claw_raw) == 0 else 1
                print(f"Gripper state: raw={claw_raw} -> {'open' if claw_status == 0 else 'closed'} (training={claw_status})")
            else:
                print(f"Warning: gripper read failed ({claw_raw}); falling back to the cached value")
                claw_status = self.claw_status
                if claw_status is not None:
                    claw_status = 0 if int(claw_status) == 0 else 1
                else:
                    print("Error: no gripper state available, recording open (0)")
                    claw_status = 0

            position_x *= 0.001
            position_y *= 0.001
            position_z *= 0.001

            roll_rad = np.radians(roll)
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)

            euler_ = [roll_rad, pitch_rad, yaw_rad]
            quaternion = transforms3d.euler.euler2quat(*euler_, axes='sxyz')
            print('Converted quaternion:', quaternion)
            print('Converted Euler angles:', euler_)

            pose_data = [position_x, position_y, position_z, *quaternion, claw_status]

        # save_counter is the single step index, keeping actions / images / depth / point clouds aligned.
            index = self.save_counter

        # --- Transactional save, phase 1: grab every enabled camera frame first ---
        # Any grab failure aborts the step before anything is written, so actions can never have step N while
        # one camera is missing that frame. The wrist camera is off by default (rs_image is None).
            rs_image = self._capture_realsense_frame()
            zed_result = self._capture_zed_frame()

        except Exception as e:
            print(f"This step failed and no file was written (save_counter not incremented): {e}")
            return

        # --- Transactional save, phase 2: write everything, verify integrity, roll the whole step back on failure ---
        try:
            self._write_step_files(index, pose_data, rs_image, zed_result)
            missing = [p for p in self._step_file_paths(index) if not os.path.exists(p)]
            if missing:
                raise RuntimeError(f"write verification failed, missing files: {missing}")
        except Exception as e:
            print(f"Error saving data and images: {e}")
            removed = self._remove_step_files(index)
            self._notify(
                f"the whole step was rolled back ({removed} file(s) cleaned up, save_counter not incremented); press 'a' again.",
                hud=f"Save FAILED, step {index} rolled back",
            )
            return

        self.save_counter += 1
        self._notify(
            f"data point {self.save_counter} saved (step index {index}, {len(self.STEP_FILES)} files verified).",
            hud=f"Saved step {index} (total {self.save_counter})",
        )

    def _capture_realsense_frame(self):
        """Grab one wrist RealSense colour frame; raises on failure so the caller drops the whole step.
        Returns None when the wrist camera is disabled, and no wrist_cam_* file is written for the step."""
        if not self.use_wrist:
            return None
        if not self.realsense_camera:
            raise RuntimeError("RealSense camera not initialized")
        pipeline = self.realsense_camera['pipeline']
        # Flush stale frames queued in the pipeline, so we never save the view from before the keypress.
        for _ in range(8):
            if not pipeline.poll_for_frames():
                break
        frames = pipeline.wait_for_frames(5000)
        rgb_frame = frames.get_color_frame()
        if not rgb_frame:
            raise RuntimeError("RealSense returned no color frame")
        return np.asanyarray(rgb_frame.get_data())

    def _capture_zed_frame(self):
        """Grab one ZED RGB/depth/point-cloud frame; raises on failure so the caller drops the whole step."""
        if not self.zed_camera:
            raise RuntimeError("ZED camera not initialized")
        with self.zed_lock:
            return self.zed_camera.capture()

    def _write_step_files(self, index, pose_data, rs_image, zed_result):
        """Write every file of one time step. Filenames, fields and formats match the original script exactly;
        any write failure raises so the caller can roll the whole step back."""
        pkl_filepath = os.path.join(self.actions_folder, f"{index}.pkl")
        with open(pkl_filepath, 'wb') as pkl_file:
            pickle.dump(np.array(pose_data), pkl_file)

        # wrist_cam_imgs/{i}.png + wrist_cam_rgb/{i}.pkl (raw RealSense BGR). Written only with --use_wrist;
        # otherwise those directories never exist and self.STEP_FILES omits them.
        if self.use_wrist:
            save_path = os.path.join(self.folder_path, "wrist_cam_imgs")
            os.makedirs(save_path, exist_ok=True)
            image_path = os.path.join(save_path, f"{index}.png")
            if not cv2.imwrite(image_path, rs_image):
                raise RuntimeError(f"cv2.imwrite failed: {image_path}")
            print(f"RealSense RGB image saved as PNG to: {image_path}")

            rgb_pkl_folder = os.path.join(self.folder_path, "wrist_cam_rgb")
            os.makedirs(rgb_pkl_folder, exist_ok=True)
            rgb_pkl_path = os.path.join(rgb_pkl_folder, f"{index}.pkl")
            with open(rgb_pkl_path, 'wb') as rgb_pkl_file:
                pickle.dump(rs_image, rgb_pkl_file)
            print(f"RealSense RGB image converted and saved to PKL: {rgb_pkl_path}")

        # 3rd_cam_imgs/{i}.png + 3rd_cam_rgb/{i}.pkl — as in the original script, the ZED frame is channel-
        # swapped once and the same array is written both as PNG (via PIL) and as PKL.
        zed_rgb = cv2.cvtColor(np.ascontiguousarray(zed_result["rgb"]), cv2.COLOR_RGB2BGR)
        save_path = os.path.join(self.folder_path, "3rd_cam_imgs")
        os.makedirs(save_path, exist_ok=True)
        image_path_rgb = os.path.join(save_path, f"{index}.png")
        Image.fromarray(zed_rgb).save(image_path_rgb)
        print(f"ZED RGB image saved to: {image_path_rgb}")

        rgb_pkl_folder = os.path.join(self.folder_path, "3rd_cam_rgb")
        os.makedirs(rgb_pkl_folder, exist_ok=True)
        rgb_pkl_path = os.path.join(rgb_pkl_folder, f"{index}.pkl")
        with open(rgb_pkl_path, 'wb') as rgb_pkl_file:
            pickle.dump(zed_rgb, rgb_pkl_file)
        print(f"Zed RGB image converted and saved to PKL: {rgb_pkl_path}")

        # 3rd_cam_depth/{i}.npy — uint16 depth, DEPTH_SCALE units per metre.
        # Replaces the old 3rd_cam_depth/{i}.pkl + 3rd_cam_pcd/{i}.pkl pair: the XYZ buffer's X/Y channels are
        # exactly (u-cx)/fx*Z and (v-cy)/fy*Z, so storing them cost 24.9 MB/frame to carry zero information.
        # The trainer deprojects using the intrinsics in intrinsic.pkl instead: 33.2 MB -> 4.1 MB per frame and
        # ~48 ms -> ~5 ms per write (uncompressed on purpose; PNG encoding belongs in the offline organize pass).
        # Z comes from the point cloud rather than MEASURE.DEPTH, so the reconstruction is exact by construction.
        save_path_depth = os.path.join(self.folder_path, "3rd_cam_depth")
        os.makedirs(save_path_depth, exist_ok=True)
        depth_npy_path = os.path.join(save_path_depth, f"{index}.npy")
        z_m = zed_result["pcd"][..., 2]
        z_m = np.where(np.isfinite(z_m), z_m, 0.0)
        d16 = np.clip(np.rint(z_m * DEPTH_SCALE), 0, DEPTH_MAX_U16).astype(np.uint16)
        np.save(depth_npy_path, d16)
        print(f"ZED depth saved to: {depth_npy_path}")
    def register_callback(self, callback):
        self.callbacks.append(callback)
        return len(self.callbacks) - 1

    def _cleanup_empty_episode(self):
        """On exit, remove the current episode directory when it never saved a step.

        It is only removed when it holds nothing but this script's auto-generated metadata (instruction /
        intrinsics / extrinsics); any data or unknown file is left untouched. This keeps empty {slug}_{N}
        directories from consuming a number and being treated as broken episodes downstream.
        """
        if self.save_counter > 0:
            return
        folder = self.folder_path
        step_dirs = {subdir for subdir, _ in self.STEP_FILES}
        known_files = {"instruction.txt", "instruction.pkl", "extrinsic_matrix.npy", "intrinsic.pkl"}
        try:
            entries = os.listdir(folder)
        except OSError:
            return
        for name in entries:
            path = os.path.join(folder, name)
            if name in step_dirs and os.path.isdir(path):
                if os.listdir(path):  # a non-empty modality directory means real data — keep it
                    return
            elif name not in known_files:
                return  # unknown file: keep it, to be safe
        try:
            shutil.rmtree(folder)
            print(f"removed the empty episode directory (nothing was saved): {folder}")
        except OSError as e:
            print(f"failed to remove the empty episode directory: {e}")

    def _print_session_summary(self):
        """Print the session summary on exit: where data landed, how many steps, and feed health."""
        s = self._stats
        print("=" * 60)
        print("Collection session ended")
        print(f"  current episode: {self.folder_path}")
        print(f"  steps saved in this episode: {self.save_counter}")
        print(f"  feedback stream: {s['packets_received']} packets, {s['invalid_packets']} invalid, "
              f"{s['resyncs']} resyncs, {s['dropped_bytes']} bytes dropped while aligning")
        if s['invalid_packets'] or s['resyncs']:
            print("  [note] the feed went out of alignment during this session — usually network jitter or another")
            print("         process contending for the arm. The checksum caught it; if it recurs, check the cabling and other processes.")
        trash_root = os.path.join(self.data_root, self.UNDO_TRASH_DIRNAME)
        if os.path.isdir(trash_root):
            print(f"  trash bin (data undone with 'b' is recoverable here): {trash_root}")
        print("CR5 data collection has stopped, all cameras have been closed.")
        print("=" * 60)

    def stop(self):
        """Stop every thread and release every resource (idempotent: ESC / Ctrl+C / exceptions all land here)"""
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self.display_running = False

        # Wait for any in-flight a/b/c operation to finish before closing cameras/sockets, so no half-step is written.
        if self.save_worker_lock.acquire(timeout=8.0):
            self.save_worker_lock.release()
        else:
            print("Warning: an operation was still in progress at shutdown; last step may be incomplete.")

        # Let the gripper-poll thread exit before Dashboard cleanup, so cleanup commands don't interleave with polls.
        if self.claw_thread is not None and self.claw_thread.is_alive():
            self.claw_thread.join(timeout=2.0)

        if hasattr(self, 'sock') and self.sock:
            self.sock.close()

        # Close the Modbus channel and clear errors / leave drag mode before closing Dashboard, so the next
        # startup does not find a stale channel (ModbusCreate -1) or a stuck Backdrive (Tool() -1).
        if self.dashboard_sock is not None:
            try:
                self._close_modbus_channels()
                print("Cleanup: Modbus channels closed")
            except Exception as e:
                print(f"Cleanup close modbus failed: {e}")
            try:
                resp = self._send_dashboard_command("ClearError()")
                print(f"Cleanup ClearError -> {resp}")
            except Exception as e:
                print(f"Cleanup ClearError failed: {e}")
            try:
                self.dashboard_sock.close()
            except Exception:
                pass
            self.dashboard_sock = None
        
        if self.realsense_camera:
            self.realsense_camera['pipeline'].stop()
        if self.zed_camera:
            self.zed_camera.stop()
            
        if self.key_listener:
            self.key_listener.stop()
            
        if hasattr(self, 'recv_thread') and self.recv_thread.is_alive():
            self.recv_thread.join(timeout=1.0)
            
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
            
        if hasattr(self, 'display_thread') and self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
            
        if self.display:
            cv2.destroyAllWindows()

        # Clean up an empty episode (no step saved) and print the session summary.
        self._cleanup_empty_episode()
        self._print_session_summary()

        # Release the single-instance lock explicitly, so a restart works immediately.
        if self._instance_lock_sock is not None:
            try:
                self._instance_lock_sock.close()
            except Exception:
                pass
            self._instance_lock_sock = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CR5 robotic arm real-time data monitoring and camera image acquisition '
                    '(ZED third-person view by default; add --use_wrist for the RealSense wrist camera)')
    parser.add_argument('--ip', default=DEFAULT_ARM_IP,
                        help='Robot arm IP address (default: $ARM_IP, else a placeholder '
                             'you are expected to replace)')
    parser.add_argument('--port', type=int, default=30004, help='Data port (30004/30005/30006)')
    parser.add_argument('--freq', type=int, default=10, help='Data processing frequency (Hz), default to 10Hz')
    parser.add_argument(
        '--use_wrist', action='store_true',
        help='Also record the wrist RealSense camera (wrist_cam_imgs/, wrist_cam_rgb/). '
             'Off by default: the downstream pipeline only consumes 3rd_cam_* + actions.',
    )
    parser.add_argument('--realsense_serial', default=DEFAULT_REALSENSE_SERIAL,
                        help='Wrist RealSense serial number, $REALSENSE_SERIAL '
                             '(only needed with --use_wrist)')
    parser.add_argument('--zed_serial', default=DEFAULT_ZED_SERIAL,
                        help='ZED camera serial number, $ZED_3RD_SERIAL; '
                             '0 = the first camera the SDK reports')
    parser.add_argument('--extrinsics_file', default=DEFAULT_EXTRINSICS_FILE,
                        help='Extrinsics matrix file path (npy). '
                             'Default: $REAL_CAM_EXTRINSICS — required, no built-in path.')
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT,
                        help='Root dir episodes are written to, as {root}/{slug}_{N}/. '
                             'Default: $REAL_COLLECT_DATA_ROOT — required, no built-in path.')
    parser.add_argument(
        '--instruction',
        default=None,
        help='Override language instruction (default: INSTRUCTION below)',
    )
    parser.add_argument('--display', action='store_true', help='Enable live image display')
    args = parser.parse_args()
    if args.use_wrist and not args.realsense_serial:
        parser.error("--use_wrist needs a wrist camera serial: pass "
                     "--realsense_serial or set $REALSENSE_SERIAL.")

    # Language instruction for this collection session; the episode folder slug
    # and instruction.txt are both derived from it. --instruction overrides it.
    INSTRUCTION = "Press the blue button twice, then press the yellow button"

    instruction = args.instruction if args.instruction is not None else INSTRUCTION

    # Data root ($REAL_COLLECT_DATA_ROOT or --data_root); episodes land in {data_root}/{slug}_{N}/. Missing
    # values abort here rather than silently collecting somewhere else. The extrinsics file is validated too,
    # so a bad calibration path is caught before the cameras and arm finish initialising.
    data_root = require_data_root(args.data_root, must_exist=False)
    require_file(args.extrinsics_file, "REAL_CAM_EXTRINSICS", "--extrinsics_file",
                 "camera extrinsics file (.npy)")
    os.makedirs(data_root, exist_ok=True)

    cr5 = None
    try:
        cr5 = CR5Realtime(
            ip=args.ip,
            port=args.port,
            frequency=args.freq,
            data_root=data_root,
            realsense_serial=args.realsense_serial,
            zed_serial=int(args.zed_serial),
            instruction=instruction,
            extrinsics_file=args.extrinsics_file,
            display=args.display,
            use_wrist=args.use_wrist,
        )

        cr5.register_callback(cr5.pose_callback)
        cr5.start()
        cr5.start_keyboard_listener()
        
        cams = "ZED(3rd_cam) + RealSense(wrist_cam)" if args.use_wrist else "ZED(3rd_cam) only [--use_wrist adds the wrist camera]"
        print(f"Cameras enabled: {cams}")
        CR5Realtime._print_controls_help()

        while cr5.running:
            time.sleep(0.1)

    except ConnectionRefusedError:
        print("\nConnection refused! Please check the following steps:")
        print("1. Ensure the robot arm is powered and the network cable is connected")
        print("2. Ensure the robot arm is enabled (via the teach pendant)")
        print("3. Try connecting to other ports: 30005 (200ms) or 30006 (50ms)")
        print(f"4. Check if the computer's IP is in the same subnet as the robot ({args.ip})\n")
    except KeyboardInterrupt:
        print("\nUser interrupt, closing...")
    finally:
        # Every exit path (ESC / Ctrl+C / exception / normal end) goes through stop(), which is idempotent.
        # ESC used to bypass it, leaving the Modbus channel registered in the controller (next ModbusCreate
        # returns -1) and the camera unreleased.
        if cr5 is not None:
            cr5.stop()
