import sys
import os
import glob
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH, ASSETS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *


import sys
import os
import subprocess
import socket
import json
import threading
import time
import random
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import json
from typing import Any

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

import numpy as np
import json
from typing import Any
import base64

class NumpyEncoder(json.JSONEncoder):
    """Enhanced json encoder for numpy types with array reconstruction info"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float32:
                dtype = 'float32'
            elif obj.dtype == np.float64:
                dtype = 'float64'
            elif obj.dtype == np.int32:
                dtype = 'int32'
            elif obj.dtype == np.int64:
                dtype = 'int64'
            else:
                dtype = str(obj.dtype)
            
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': dtype,
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    """Convert numpy-containing data to JSON string with reconstruction info"""
    return json.dumps(data, cls=NumpyEncoder)

def json_to_numpy(json_str: str) -> Any:
    """Convert JSON string back to Python objects with numpy arrays"""
    def object_hook(dct):
        if '__numpy_array__' in dct:
            data = base64.b64decode(dct['data'])
            return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    
    return json.loads(json_str, object_hook=object_hook)

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    # conda_env is abandoned
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def _env_flag_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_flag01(name: str, default: str = "0") -> int:
    return 1 if _env_flag_on(name, default) else 0


RESULT_EPISODE_HEADER = "==== Per-episode results ===="


def _init_result_file(file_path: str, timestamp: str, instruction_type: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Instruction Type: {instruction_type}\n\n")
        f.write(f"{RESULT_EPISODE_HEADER}\n")


def _append_episode_result(file_path: str, rec: dict) -> None:
    tag = "success" if rec["success"] else "fail"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(
            f"episode{rec['episode']}\t{tag}\tseed={rec['seed']}\t"
            f"{rec['instruction']}\n"
        )


def _finalize_result_file(file_path: str, success_rate: float) -> None:
    """Insert the final success rate between the Instruction Type block and the per-episode details."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    marker = f"\n\n{RESULT_EPISODE_HEADER}\n"
    if marker not in text:
        raise ValueError(f"Malformed result file (missing episode header): {file_path}")
    header_part, rest = text.split(marker, 1)
    final = (
        header_part.rstrip("\n")
        + f"\n\n{success_rate}\n\n{RESULT_EPISODE_HEADER}\n"
        + rest
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

class ModelClient:
    def __init__(self, host='localhost', port=9999, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = 1000
        retry_delay = 5
        
        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {str(e)}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to server after {max_attempts} attempts: {str(e)}"
                    )

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Reconnect fallback: a previous communication failure (e.g. a timeout) closes the socket and
            # sets it to None; the server's accept loop is permanent, so reconnect here and send this
            # request as usual rather than letting one glitch kill the whole eval run.
            if self.sock is None:
                print("⚠️ Socket was closed by a previous error; reconnecting...")
                self._connect()

            # Serialize with numpy support
            json_data = numpy_to_json(data).encode('utf-8')
            
            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, 'big'))
            self.sock.sendall(json_data)
            
            # Receive and deserialize response
            response = self._recv_response()
            return response
            
        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        len_data = self.sock.recv(4)
        if not len_data:
            raise ConnectionError("Connection closed by server")
        
        size = int.from_bytes(len_data, 'big')
        
        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)
        
        # Deserialize with numpy reconstruction
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        return response['res']

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main(usr_args):
    # Timestamps are joined with underscores (no spaces/colons/dashes in directory names) and carry no year, e.g. 06_11_11_50_13
    current_time = datetime.now().strftime("%m_%d_%H_%M_%S")
    force_center_flag = _env_flag01("stage2_force_center", "0")
    sparse_pc_force_center_flag = _env_flag01("stage2_sparsePC_force_center", "0")
    run_dir_name = (
        f"{current_time}_force_center_{force_center_flag}"
        f"_sparsePC_force_center_{sparse_pc_force_center_flag}"
    )
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    port = usr_args["port"]
    save_dir = None
    video_save_dir = None
    video_size = None

    policy_conda_env = usr_args.get("policy_conda_env", None)

    get_model = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    # policy/<policy_name>/eval_config.yml: collects the "record or not / keyframe vs continuous / max steps
    # per task / test_num" knobs into the policy directory (next to eval_double_env.sh), overriding the
    # task_config defaults so you never have to dig through the shared demo_clean.yml and
    # _eval_step_limit.yml. Any entry left null overrides nothing and keeps the task_config default.
    # eval_double_env.sh snapshots it into RMBENCH_EVAL_CONFIG_SNAPSHOT at startup, and every task in the
    # run reads only the snapshot (editing eval_config.yml on disk mid-run has no effect).
    # Priority: environment variables (the RMBENCH_EVAL_* below) > this file > the task_config defaults.
    eval_cfg = {}
    eval_cfg_path = os.environ.get("RMBENCH_EVAL_CONFIG_SNAPSHOT") or os.path.join(
        "./policy", policy_name, "eval_config.yml"
    )
    if os.path.isfile(eval_cfg_path):
        with open(eval_cfg_path, "r", encoding="utf-8") as f:
            eval_cfg = yaml.safe_load(f) or {}
        print(f"\033[36m[eval] loaded eval_config -> {eval_cfg_path}\033[0m")
        for key in ("eval_video_log", "eval_video_mode",
                    "eval_video_step_freq", "eval_video_fps"):
            if eval_cfg.get(key) is not None:
                args[key] = eval_cfg[key]
        # Max steps per task: the whole table is passed to the environment, and _base_task looks up
        # task_name; tasks left null or missing fall back to task_config/_eval_step_limit.yml.
        if isinstance(eval_cfg.get("step_limit"), dict):
            args["eval_step_limit_override"] = eval_cfg["step_limit"]

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        # file_path historically read "./assets/embodiments/..." relative to finetune/RMBench/, resolved
        # through the data/assets symlink. That symlink is gone and assets moved to the data root, so the
        # assets prefix is stripped here and the rest anchored to ASSETS_PATH, returning an absolute path
        # (as in collect_data.py, accepting "./assets/...", "assets/..." and "embodiments/...").
        rel = os.path.normpath(robot_file).split(os.sep)
        if rel and rel[0] == "assets":
            rel = rel[1:]
        return os.path.join(ASSETS_PATH, *rel)

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # Eval results land under the training run directory (mirroring memoryBench's MODEL_FOLDER/eval/<benchmark>/...).
    # In BridgeVLA, ckpt_setting *is* the training run dir (holding model_*.pth + exp_cfg.yaml +
    # mvt_cfg.yaml), the equivalent of memoryBench's MODEL_FOLDER. The innermost directory name = timestamp
    # (no year) + stage-2 flags; the model name (e.g. model_105) stays a separate level, so different eval strategies on one ckpt can be compared:
    #   * ckpt_setting is a directory -> <run_dir>/eval/rmbench/<task>/<cfg>/<model_name>/<ts>_<flags>/
    #     (the model name is resolved by the same rule as the server: model_<model_epoch>.pth, falling back to model_last.pth)
    #   * ckpt_setting is a .pth file  -> dirname(.pth) is the run dir and the model name = the .pth filename without its suffix
    #   * neither a directory nor a file (a bare tag) -> fall back to the old relative eval_result/ as a safety net
    # flags look like force_center_0_sparsePC_force_center_1 (always an explicit 0/1, both 0 by default)
    run_dir = None
    model_name = None
    if isinstance(ckpt_setting, str) and ckpt_setting:
        if os.path.isdir(ckpt_setting):
            run_dir = ckpt_setting
            model_epoch = usr_args.get("model_epoch", "last")
            cand = os.path.join(run_dir, f"model_{model_epoch}.pth")
            if not os.path.isfile(cand):
                cand = os.path.join(run_dir, "model_last.pth")
            if not os.path.isfile(cand):
                # A released ckpt directory usually holds a single weight (with the epoch in its name) — keep
                # this in sync with BridgeVLAModelServer's resolution on the server side, or the result directory would be labelled with the wrong model name.
                only = sorted(glob.glob(os.path.join(run_dir, "model_*.pth")))
                if len(only) == 1:
                    cand = only[0]
            model_name = Path(cand).stem
        elif os.path.isfile(ckpt_setting):
            run_dir = os.path.dirname(os.path.abspath(ckpt_setting))
            model_name = Path(ckpt_setting).stem
    if run_dir is not None:
        save_dir = Path(run_dir) / "eval" / "rmbench" / task_name / task_config / model_name / run_dir_name
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{run_dir_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\033[36m[eval] results -> {save_dir}\033[0m")

    # RMBENCH_EVAL_VIDEO=1/0 turns recording on/off temporarily (higher priority than eval_config.yml;
    # same semantics as RMBENCH_EVAL_OVERLAY_VIZ). Unset (the default) keeps eval_config.yml.
    _video_env = os.environ.get("RMBENCH_EVAL_VIDEO")
    if _video_env is not None:
        args["eval_video_log"] = (_video_env == "1")
    # RMBENCH_EVAL_VIDEO_MODE=keyframe|smooth temporarily overrides the keyframe/continuous mode.
    if os.environ.get("RMBENCH_EVAL_VIDEO_MODE"):
        args["eval_video_mode"] = os.environ["RMBENCH_EVAL_VIDEO_MODE"]

    # The third_view camera exists only for recording, and eval_video_log is its sole master switch.
    # Disabling recording => the whole third_view rendering chain is disabled too (get_obs no longer calls
    # take_picture on third_view), saving a camera that costs ~20s per frame under Orion's software Vulkan.
    # The BridgeVLA model only uses head/front/left/right (see deploy_policy.encode_obs) and never consumes
    # third_view, so disabling it is lossless for the policy. Overlay viz does not depend on third_view (below), hence only eval_video_log matters here.
    if not args.get("eval_video_log"):
        if isinstance(args.get("data_type"), dict):
            args["data_type"]["third_view"] = False

    # ---- Overlay heatmap visualisation switch (eval_config.yml: eval_overlay_viz). ----
    # It only controls whether the per-step predicted-heatmap overlays of the three views (per arm, per
    # stage) are saved; the server stitches them into a grid at the end of the episode. All of them come
    # from the network forward + point-cloud rendering already computed when producing the action (the
    # top/front/right network views) and never touch the third_view camera, so they are fully decoupled
    # from recording: with recording off and third_view unrendered these images are still produced, and
    # vice versa. Output lands in <save_dir>/viz/.
    # Priority: the RMBENCH_EVAL_OVERLAY_VIZ (1/0) environment variable > eval_config.yml.
    overlay_viz = bool(eval_cfg.get("eval_overlay_viz"))
    _ov_env = os.environ.get("RMBENCH_EVAL_OVERLAY_VIZ")
    if _ov_env is not None:
        overlay_viz = (_ov_env == "1")
    args["eval_overlay_viz"] = overlay_viz
    if overlay_viz:
        viz_dir = save_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        args["eval_overlay_viz_dir"] = str(viz_dir)
        print(f"\033[36m[eval] overlay viz ON -> {viz_dir}\033[0m")

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir
        # When recording is on, the per-step diagnostic txt files (gripper / zoom / action_pose /
        # plan_status) are written too, sharing the viz/episode*/step_* layout with overlay viz; without overlay no heatmaps are rendered.
        if not overlay_viz:
            step_diag_dir = save_dir / "viz"
            step_diag_dir.mkdir(parents=True, exist_ok=True)
            args["eval_step_diag_dir"] = str(step_diag_dir)
        args["eval_step_diag"] = True

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    # Optional: start from a given sim seed (to reproduce a single episode) instead of counting up from st_seed.
    # Priority: RMBENCH_EVAL_START_SEED > eval_config.yml start_seed > the st_seed default.
    _start_seed = eval_cfg.get("start_seed")
    if os.environ.get("RMBENCH_EVAL_START_SEED"):
        _start_seed = os.environ["RMBENCH_EVAL_START_SEED"]
    if _start_seed is not None and str(_start_seed).strip() != "":
        st_seed = int(_start_seed)
        print(f"\033[36m[eval] start_seed override -> {st_seed}\033[0m")
    suc_nums = []
    # 100 by default; eval_config.yml's test_num changes that default, and RMBENCH_EVAL_TEST_NUM
    # overrides it temporarily (highest priority). Lower it for a quick pipeline check (each episode takes
    # tens of minutes under Orion's software Vulkan, so a full 100 seeds is impractical).
    test_num_default = eval_cfg.get("test_num") if eval_cfg.get("test_num") is not None else 100
    test_num = int(os.environ.get("RMBENCH_EVAL_TEST_NUM", str(test_num_default)))
    topk = 1

    file_path = os.path.join(save_dir, "_result.txt")
    _init_result_file(file_path, current_time, instruction_type)

    model = ModelClient(port=port)
    st_seed, suc_num, episode_records = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   policy_conda_env=policy_conda_env,
                                   result_file_path=file_path)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    success_rate = float(np.array(suc_nums) / test_num)
    _finalize_result_file(file_path, success_rate)

    print(f"Data has been saved to {file_path}")


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None,
                result_file_path=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    # Per-episode log (episode idx, success/fail, seed, language instruction) so
    # _result.txt can list per-episode outcomes + instructions, not just the aggregate success rate.
    episode_records = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        # Pass the overlay / step-diag switches to this episode's env (setup_demo rebuilds TASK_ENV per
        # episode, so they are refreshed every round). deploy_policy.eval reads these attributes to decide
        # whether get_action carries a viz / diag payload and whether scene frames are saved.
        TASK_ENV.eval_overlay_viz = args.get("eval_overlay_viz", False)
        TASK_ENV.eval_overlay_viz_dir = args.get("eval_overlay_viz_dir")
        TASK_ENV.eval_step_diag = args.get("eval_step_diag", False)
        TASK_ENV.eval_step_diag_dir = args.get("eval_step_diag_dir")

        if TASK_ENV.eval_video_path is not None:
            # In keyframe mode there is 1 frame per keyframe, so 10fps is fine; in smooth mode each keyframe
            # contains many physics-step samples and needs a higher fps to avoid slow motion. eval_video_fps overrides the default.
            video_fps = args.get("eval_video_fps")
            if video_fps is None:
                video_fps = 30 if args.get("eval_video_mode", "keyframe") == "smooth" else 10
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    str(video_fps),
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        model.call(func_name='reset_model')
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        # Overlay visualisation: once the episode ends, ask the server to stitch this episode's per-step
        # three-view overlays into a per-(arm, stage) grid (stitched server-side; the client env has no matplotlib).
        if args.get("eval_overlay_viz", False):
            try:
                model.call(func_name="finalize_episode_viz")
            except Exception as e:
                print(f"[RMBench eval viz] finalize_episode_viz failed: {e}")

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Record this episode's outcome + seed + language instruction (the episode index matches the
        # episode{test_num} directory used by overlay viz).
        rec = {
            "episode": int(TASK_ENV.test_num),
            "success": bool(succ),
            "seed": int(now_seed),
            "instruction": instruction,
        }
        episode_records.append(rec)
        if result_file_path:
            _append_episode_result(result_file_path, rec)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, episode_records


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config['port'] = args.port

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
