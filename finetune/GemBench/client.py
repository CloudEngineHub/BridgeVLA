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
Adapted from https://github.com/vlc-robot/robot-3dlotus/blob/main/challenges/client.py

'''
import os
import argparse
import requests
import random
import jsonlines
from tqdm import tqdm
import msgpack_numpy
msgpack_numpy.patch()

from rlbench.backend.utils import task_file_to_task_class
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

from genrobo3d.rlbench.environments import RLBenchEnv, Mover
from pyrep.objects import VisionSensor, Dummy
from genrobo3d.rlbench.recorder import (
    TaskRecorder, StaticCameraMotion, CircleCameraMotion,
)
import warnings
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
            f"[GemBench client] cannot reach the server at {server_addr} "
            f"to verify memory switches: {e}"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"[GemBench client] server {server_addr} has no /memory_config "
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
            "[GemBench client] memory switches do NOT match the server:\n"
            + "\n".join(mismatches) + "\n"
            "Fix TEMPORAL_MEMORY / SPATIAL_MEMORY in run_client.sh to match "
            "run_server.sh (which was validated against the loaded model)."
        )
    print(f"[GemBench client] memory switches OK "
          f"(temporal={server_temporal}, spatial={server_spatial}).")


def main(taskvar, server_addr, microstep_data_dir='',output_file=None,record_video=False,video_rotate_cam=False,video_resolution_width=320,video_resolution_height=180,
         visualize=False, visualize_root_dir="", max_episodes=0,
         expect_temporal_memory=True, expect_spatial_memory=True):
    NUM_EPISODES = 25
    MAX_STEPS = 25
    IMAGE_SIZE = 256

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

    episode_id = 0
    step_id = 0
   
    env = RLBenchEnv(
        data_path=microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=False,
        headless=True,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
        cam_rand_factor=0,
    )

    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task.set_variation(variation_id)


    if record_video:
        # Add a global camera to the scene
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

    if microstep_data_dir != '':
        episodes_dir = os.path.join(microstep_data_dir, task_str, f"variation{variation_id}", "episodes")
        demos = []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                try:
                    demo = env.get_demo(task_str, variation_id, idx, load_images=False)
                    demos.append(demo)
                except Exception as e:
                    print('\tProblem to load demo_id:', idx, ep)
                    print(e)
        NUM_EPISODES = len(demos)
    else:
        demos = None

    if max_episodes and max_episodes > 0:
        NUM_EPISODES = min(NUM_EPISODES, max_episodes)
        if demos is not None:
            demos = demos[:NUM_EPISODES]

    success_rate = 0

    for episode_id in tqdm(range(NUM_EPISODES)):
        if demos is None:
            instructions, obs = task.reset()
        else:
            print("Resetting to demo", episode_id)
            instructions, obs = task.reset_to_demo(demos[episode_id])  # type: ignore

        instruction = random.choice(instructions)
        print(instruction)

        obs_state_dict = env.get_observation(obs)
        move.reset(obs_state_dict['gripper'])

        if visualize:
            episode_viz_dir = os.path.join(taskvar_viz_dir, f"episode_{episode_id}")
            os.makedirs(episode_viz_dir, exist_ok=True)
        else:
            episode_viz_dir = ""

        for step_id in range(MAX_STEPS):
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
            action = msgpack_numpy.unpackb(response._content)

            if action is None:
                break

            # update the observation based on the predicted action
            try:
                obs, reward, terminate, _ = move(action, verbose=False)
                error_type = None
                obs_state_dict = env.get_observation(obs)  # type: ignore
                if reward == 1:
                    success_rate += 1 / NUM_EPISODES
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
        # of which live server-side). No-op server-side when viz is off.
        if visualize:
            try:
                requests.post(
                    f"{server_addr}/finalize",
                    data=msgpack_numpy.packb({}, use_bin_type=True),
                )
            except Exception as e:
                print(f"[GemBench eval viz] finalize request failed: {e}")

        if record_video:
            tr.save(os.path.join(video_log_dir, f"{episode_id}_SR{reward}"))

        with jsonlines.open(output_file, 'a', flush=True) as outf:
            outf.write({
                'episode_id': episode_id,
                'instr': instruction, 
                'success': reward,
                'error': error_type,
                'nsteps': step_id+1,
            })

    print('Success Rate: {:.2f}%'.format(success_rate*100))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--taskvar', default='close_microwave2+0')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13003)
    parser.add_argument('--output_file', type=str, default="PATH_TO_SAVE_RESULT_JSON/result.json")
    parser.add_argument('--microstep_data_dir', default='/PATH_TO_GEMBENCH_TEST_DATA/microsteps/seed300')
    parser.add_argument('--record_video', action='store_true', help='Record mp4 video of each episode')
    parser.add_argument('--video_rotate_cam', action='store_true', help='Rotating cinematic camera')
    parser.add_argument('--video_resolution_width', type=int, default=320)
    parser.add_argument('--video_resolution_height', type=int, default=180)
    parser.add_argument('--visualize', action='store_true', help='Save per-step rendered views, heatmaps, pred-waypoint overlay, and point cloud')
    parser.add_argument('--visualize_root_dir', type=str, default="", help='Root dir for visualize outputs (required if --visualize)')
    parser.add_argument('--max_episodes', type=int, default=0, help='Cap episodes per taskvar (0 = no cap, use all available demos)')
    # Memory ablation switches (client/server alignment): must match the
    # server's --temporal_memory / --spatial_memory or the client aborts at
    # startup. Set via run_client.sh TEMPORAL_MEMORY / SPATIAL_MEMORY.
    parser.add_argument('--temporal_memory', type=str, default="true",
                        help='true/false; must match the server (stage-1 temporal memory)')
    parser.add_argument('--spatial_memory', type=str, default="true",
                        help='true/false; must match the server (stage-2 spatial anchor memory)')
    args = parser.parse_args()
    server_addr = f"http://{args.ip}:{args.port}/"
    main(
        args.taskvar,
        server_addr,
        args.microstep_data_dir,
        args.output_file,
        record_video=args.record_video,
        video_rotate_cam=args.video_rotate_cam,
        video_resolution_width=args.video_resolution_width,
        video_resolution_height=args.video_resolution_height,
        visualize=args.visualize,
        visualize_root_dir=args.visualize_root_dir,
        max_episodes=args.max_episodes,
        expect_temporal_memory=parse_bool_flag(
            args.temporal_memory, "--temporal_memory"),
        expect_spatial_memory=parse_bool_flag(
            args.spatial_memory, "--spatial_memory"),
    )
