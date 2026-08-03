"""
MemoryBench rollout client.

Sends per-step observations to the MyActioner server and steps the RLBench
simulator with the predicted actions. Adapted from finetune/GemBench/client.py
with two MemoryBench-specific tweaks:

  1. Episode horizon is bumped from MAX_STEPS=25 to MAX_STEPS=50.
     MemoryBench tasks are *long-horizon by construction* -- put_block_back has
     ~10 keyframes (lift, navigate, press button, navigate back, lower, release)
     which routinely exceeds 25 motion-planner waypoint requests once retries
     are counted. 25 was tuned for GemBench's much shorter atomic tasks.

  2. Two demo-source modes:
        a) --episodes_data_dir points at the standard RLBench per-variation
           layout (<task>/variation<vid>/episodes/...). Same path as GemBench;
           we call env.env.get_demos and feed task.reset_to_demo().
        b) --episodes_data_dir omitted (or pointed at the all_variations/
           layout) -> we fall back to plain task.reset(). The simulator picks
           a random scene state each episode. Less deterministic but doesn't
           require restructuring the released MemoryBench data.

"""
import argparse
import os
import random
import warnings

import jsonlines
import numpy as np
import msgpack_numpy
msgpack_numpy.patch()
import requests
from tqdm import tqdm

from rlbench.backend.utils import task_file_to_task_class
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import Dummy, VisionSensor

from genrobo3d.rlbench.environments import Mover, RLBenchEnv
from genrobo3d.rlbench.recorder import (
    CircleCameraMotion, StaticCameraMotion, TaskRecorder,
)

warnings.filterwarnings("ignore", message="Object .* has ObjectType.LIGHT.*")

from bridgevla.utils.memory_switches import parse_bool_flag


def check_server_memory_config(server_addr, expect_temporal_memory,
                               expect_spatial_memory):
    '''Client/server alignment guard: fetch the server's memory ablation
    switches (already validated against the loaded model at server startup)
    and abort if the client's own switches disagree — evaluating with
    mismatched expectations would silently score the wrong ablation.'''
    try:
        response = requests.get(f"{server_addr}/memory_config")
    except requests.RequestException as e:
        raise RuntimeError(
            f"[MemoryBench client] cannot reach the server at {server_addr} "
            f"to verify memory switches: {e}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"[MemoryBench client] server {server_addr} has no /memory_config "
            f"route (HTTP {response.status_code}) — restart the server with "
            "the current run_server.sh so the memory switches can be verified."
        )
    server_cfg = response.json()
    server_temporal = bool(server_cfg.get("temporal_memory", True))
    server_spatial = bool(server_cfg.get("spatial_memory", True))
    mismatches = []
    if bool(expect_temporal_memory) != server_temporal:
        mismatches.append(
            f"  - temporal_memory: client={bool(expect_temporal_memory)} "
            f"vs server={server_temporal}"
        )
    if bool(expect_spatial_memory) != server_spatial:
        mismatches.append(
            f"  - spatial_memory: client={bool(expect_spatial_memory)} "
            f"vs server={server_spatial}"
        )
    if mismatches:
        raise RuntimeError(
            "[MemoryBench client] memory switches do NOT match the server:\n"
            + "\n".join(mismatches) + "\n"
            "Fix TEMPORAL_MEMORY / SPATIAL_MEMORY in run_client.sh to match "
            "run_server.sh (which was validated against the loaded model)."
        )
    print(f"[MemoryBench client] memory switches OK "
          f"(temporal={server_temporal}, spatial={server_spatial}).")


def _is_per_variation_layout(microstep_data_dir, task_str, variation_id):
    """Standard RLBench layout: <task>/variation<vid>/episodes/."""
    p = os.path.join(microstep_data_dir, task_str, f"variation{variation_id}", "episodes")
    return os.path.exists(p)


def main(taskvar, server_addr, microstep_data_dir='', output_file=None,
         record_video=False, video_rotate_cam=False, video_resolution_width=320,
         video_resolution_height=180, visualize=False, visualize_root_dir="",
         max_episodes=0, num_episodes=25, max_steps=50, image_size=128,
         expect_temporal_memory=True, expect_spatial_memory=True, seed=None):
    # Seed the client-side global RNGs. RLBench episode randomization
    # (SpawnBoundary.sample -> object placement/rotation in task.reset())
    # draws from the GLOBAL np.random stream, and the per-episode instruction
    # pick uses the global `random` module. Seeding both here makes the
    # sequence of episode initial states reproducible per (seed, taskvar) and
    # gives different seeds genuinely different episode sets.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"[MemoryBench] client RNGs (random, np.random) seeded with {seed}")

    # Fail fast BEFORE launching the (slow) RLBench env: the client's memory
    # switches must match the server's (which in turn match the loaded model).
    check_server_memory_config(
        server_addr, expect_temporal_memory, expect_spatial_memory)

    task_str, variation_id = taskvar.split('+')
    variation_id = int(variation_id)

    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    if visualize:
        assert visualize_root_dir, "--visualize requires --visualize_root_dir"
        taskvar_viz_dir = os.path.join(visualize_root_dir, taskvar)
        os.makedirs(taskvar_viz_dir, exist_ok=True)
    else:
        taskvar_viz_dir = ""

    env = RLBenchEnv(
        data_path=microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=False,
        headless=True,
        image_size=[image_size, image_size],
        cam_rand_factor=0,
    )
    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task.set_variation(variation_id)

    if record_video:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_resolution = [video_resolution_width, video_resolution_height]
        cam = VisionSensor.create(cam_resolution)
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)
        if video_rotate_cam:
            global_cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
        else:
            global_cam_motion = StaticCameraMotion(cam)
        cams_motion = {"global": global_cam_motion}
        tr = TaskRecorder(cams_motion, fps=30)
        task._scene.register_step_callback(tr.take_snap)
        log_dir = os.path.dirname(output_file)
        video_log_dir = log_dir + '/videos' + f'/{task_str}+{variation_id}'
        os.makedirs(str(video_log_dir), exist_ok=True)

    move = Mover(task, max_tries=10)

    # Demo source selection.
    demos = None
    if microstep_data_dir and _is_per_variation_layout(microstep_data_dir, task_str, variation_id):
        episodes_dir = os.path.join(
            microstep_data_dir, task_str, f"variation{variation_id}", "episodes",
        )
        episode_ids = os.listdir(episodes_dir)
        episode_ids.sort(key=lambda ep: int(ep[7:]))
        demos = []
        for idx, _ep in enumerate(episode_ids):
            try:
                demo = env.get_demo(task_str, variation_id, idx, load_images=False)
                demos.append(demo)
            except Exception as e:
                print(f"\tProblem to load demo_id: {idx} {_ep}: {e}")
        num_episodes = len(demos)

    if max_episodes and max_episodes > 0:
        num_episodes = min(num_episodes, max_episodes)
        if demos is not None:
            demos = demos[:num_episodes]

    print(f"[MemoryBench] {taskvar}: rolling out {num_episodes} episodes "
          f"({'reset_to_demo' if demos else 'task.reset()'})")

    success_rate = 0.0

    for episode_id in tqdm(range(num_episodes)):
        if demos is None:
            instructions, obs = task.reset()
        else:
            print("Resetting to demo", episode_id)
            instructions, obs = task.reset_to_demo(demos[episode_id])
        instruction = random.choice(instructions)
        print(instruction)

        obs_state_dict = env.get_observation(obs)
        move.reset(obs_state_dict['gripper'])

        if visualize:
            episode_viz_dir = os.path.join(taskvar_viz_dir, f"episode_{episode_id}")
            os.makedirs(episode_viz_dir, exist_ok=True)
        else:
            episode_viz_dir = ""

        reward = 0
        error_type = None
        step_id = 0
        for step_id in range(max_steps):
            batch = {
                'taskvar': taskvar,
                'episode_id': episode_id,
                'step_id': step_id,
                'instruction': instruction,
                'obs_state_dict': obs_state_dict,
                'visualize': visualize,
                'visualize_episode_dir': episode_viz_dir,
            }
            data = msgpack_numpy.packb(batch, use_bin_type=True)
            response = requests.post(f"{server_addr}/predict", data=data)
            if response.status_code != 200:
                # A non-200 body is Flask's error page, not a msgpack action —
                # unpacking it dies with a cryptic ExtraData error. The real
                # exception (often CUDA OOM) is printed in the SERVER terminal.
                raise RuntimeError(
                    f"[MemoryBench client] server /predict failed with HTTP "
                    f"{response.status_code}; check the server terminal for "
                    f"the real traceback. Response body (truncated):\n"
                    f"{response.text[:2000]}"
                )
            action = msgpack_numpy.unpackb(response._content)
            if action is None:
                break
            # ``action`` is 8-D [wpt(3), quat(4), grip(1)] — the actioner drops
            # the collision slot (see actioner.py). MemoryBench ignores per-step
            # collision checking by convention: the env is constructed with
            # EndEffectorPoseViaPlanning(collision_checking=False), and the stock
            # MoveArmThenGripper.action() passes no per-call ignore_collisions,
            # so the planner runs with ignore_collisions=True on every step.
            # Nothing is toggled here — this used to read action[8] and then
            # unconditionally force it to 1, which was the same outcome by a
            # noisier route.
            try:
                obs, reward, terminate, _ = move(action, verbose=False)
                error_type = None
                obs_state_dict = env.get_observation(obs)
                if reward == 1:
                    success_rate += 1 / num_episodes
                    break
                if terminate:
                    print("The episode has terminated!")
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(taskvar, episode_id, step_id, e)
                error_type = str(e)
                reward = 0
                break

        # Episode-end overlay viz: ask the server to stitch this episode's
        # per-step overlay tri-views into grid_mvt1.png / grid_mvt2.png
        # (stitching needs matplotlib + the agent's view/stage config, both
        # server-side). No-op server-side when viz is off.
        if visualize:
            try:
                requests.post(
                    f"{server_addr}/finalize",
                    data=msgpack_numpy.packb({}, use_bin_type=True),
                )
            except Exception as e:
                print(f"[MemoryBench eval viz] finalize request failed: {e}")

        if record_video:
            tr.save(os.path.join(video_log_dir, f"{episode_id}_SR{reward}"))

        with jsonlines.open(output_file, 'a', flush=True) as outf:
            outf.write({
                'episode_id': episode_id,
                'instr': instruction,
                'success': reward,
                'error': error_type,
                'nsteps': step_id + 1,
            })

    print(f'[MemoryBench] {taskvar} Success Rate: {success_rate*100:.2f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MemoryBench rollout client')
    parser.add_argument('--taskvar', default='put_block_back+0')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13003)
    parser.add_argument(
        '--output_file', type=str,
        default="memorybench_eval/result.jsonl",
    )
    parser.add_argument(
        '--microstep_data_dir', default='',
        help="Optional. Standard RLBench per-variation layout "
             "(<task>/variation<vid>/episodes/). If unset, we use task.reset() "
             "with random scene initialization for each episode.",
    )
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--video_rotate_cam', action='store_true')
    parser.add_argument('--video_resolution_width', type=int, default=320)
    parser.add_argument('--video_resolution_height', type=int, default=180)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize_root_dir', type=str, default="")
    parser.add_argument('--max_episodes', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=25)
    # MemoryBench tasks are longer than GemBench's; raise the cap.
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--image_size', type=int, default=128)
    # Memory ablation switches (client/server alignment): must match the
    # server's --temporal_memory / --spatial_memory or the client aborts at
    # startup. Set via run_client.sh TEMPORAL_MEMORY / SPATIAL_MEMORY.
    parser.add_argument('--temporal_memory', type=str, default="true",
                        help='true/false; stage-1 temporal memory (memory 1) the server model is expected to have')
    parser.add_argument('--spatial_memory', type=str, default="true",
                        help='true/false; stage-2 spatial anchor memory (memory 2) the server model is expected to have')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Seed for the client-side RNGs (python random + np.random) that '
             'drive RLBench episode randomization in task.reset(). Unset = '
             'nondeterministic episodes.',
    )
    args = parser.parse_args()
    server_addr = f"http://{args.ip}:{args.port}/"
    main(
        args.taskvar, server_addr, args.microstep_data_dir, args.output_file,
        record_video=args.record_video,
        video_rotate_cam=args.video_rotate_cam,
        video_resolution_width=args.video_resolution_width,
        video_resolution_height=args.video_resolution_height,
        visualize=args.visualize,
        visualize_root_dir=args.visualize_root_dir,
        max_episodes=args.max_episodes,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        image_size=args.image_size,
        expect_temporal_memory=parse_bool_flag(
            args.temporal_memory, "--temporal_memory"),
        expect_spatial_memory=parse_bool_flag(
            args.spatial_memory, "--spatial_memory"),
        seed=args.seed,
    )
