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
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/eval.py
Therefore, the code is also under the NVIDIA Source Code License

'''
import os
import yaml
import csv
import json
import torch
import cv2
import shutil
import numpy as np
import open3d as o3d
from multiprocessing import Value
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.agents.agent import VideoSummary

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.mvt.mvt import MVT
from bridgevla.libs.peract.helpers import utils
from utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
    MoveArmThenGripper2,
)
from bridgevla.utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    RLBENCH_TASKS,
)
from bridgevla.utils.rvt_utils import load_agent as load_agent_state
from bridgevla.utils.memory_switches import (
    parse_bool_flag,
    assert_eval_memory_matches_model,
)

# Default per-task step-limit table (mirrors RMBench task_config/_eval_step_limit.yml).
DEFAULT_STEP_LIMIT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "eval_step_limit.yml"
)


def load_step_limits(config_path):
    """Load the per-task max-step table (task_name -> int).

    Returns {} when config_path is falsy or missing, so callers fall back to
    the global --episode-length for every task. A task whose value is null in
    the yaml is skipped (also falls back to the global default). Mirrors
    RMBench's _eval_step_limit.yml semantics.
    """
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        print(f"[Warn] step-limit config not found: {config_path} "
              f"-> using global --episode-length for all tasks")
        return {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return {task: int(lim) for task, lim in data.items() if lim is not None}


def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False,
    expect_temporal_memory=True,
    expect_spatial_memory=True):
    device = f"cuda:{device}"
    assert model_path is not None

    # load exp_cfg
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    # Ckpts trained before a schema cleanup carry keys the current config no
    # longer defines (train-only blocks such as pretrain_mix); drop them with a
    # warning instead of refusing to load. See merge_from_file_drop_unknown.
    default_exp_cfg.merge_from_file_drop_unknown(
        exp_cfg,
        exp_cfg_path if exp_cfg_path is not None
        else os.path.join(model_folder, "exp_cfg.yaml"),
        where="RLBench eval exp_cfg",
    )

    # NOTE: to not use place_with_mean in evaluation
    # needed for rvt-1 but not rvt-2
    if not use_input_place_with_mean:
        # for backward compatibility
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True

    exp_cfg.freeze()


    # mvt_cfg stays a STRICT merge on purpose: it decides the architecture, and
    # rvt_utils.load_agent falls back to strict=False weight loading, so a
    # silently dropped mvt key could yield a subtly different network instead of
    # an error. An unknown key here must fail loudly.
    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path != None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

    mvt_cfg.freeze()

    # Guard: eval-side memory ablation switches (eval.sh TEMPORAL_MEMORY /
    # SPATIAL_MEMORY) must match the trained model's memory config.
    assert_eval_memory_matches_model(
        mvt_cfg, expect_temporal_memory, expect_spatial_memory,
        where="RLBench eval",
    )

    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    rvt = MVT(
        renderer_device=device,
        **mvt_cfg,
    )

    agent = bridgevla_agent.RVTAgent(
        network=rvt.to(device),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 1000)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )


    agent.build(training=False, device=device)
    # use_view_logvar branch is deferred (see bridgevla.mvt.view_logvar);
    # the agent constructor would already have raised NotImplementedError
    # if the flag were True.
    load_agent_state(model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent




def _merge_camera_pointcloud(observation, cameras):
    """Merge per-camera ``{cam}_point_cloud`` / ``{cam}_rgb`` from one step's
    observation dict into flat (N,3) xyz (meters) + (N,3) rgb (0-1 float)
    arrays. Handles both channel-first (3,H,W) and channel-last (H,W,3)
    layouts (RLBench env's channels_last flag).
    """
    pts, cols = [], []
    for cam in cameras:
        pc = observation.get(f"{cam}_point_cloud")
        rgb = observation.get(f"{cam}_rgb")
        if pc is None or rgb is None:
            continue
        pc = np.asarray(pc)
        rgb = np.asarray(rgb)
        if pc.ndim == 3 and pc.shape[0] == 3:
            pc = np.transpose(pc, (1, 2, 0))
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        pts.append(pc.reshape(-1, 3))
        cols.append(rgb.reshape(-1, 3).astype(np.float32) / 255.0)
    if not pts:
        return None, None
    return np.concatenate(pts, axis=0), np.concatenate(cols, axis=0)


def save_step_pointcloud(observation, cameras, save_path):
    """Save one rollout step's merged multi-camera point cloud (xyz+rgb) as
    a .ply file. No-op if the observation carries no point-cloud keys.
    """
    xyz, rgb = _merge_camera_pointcloud(observation, cameras)
    if xyz is None:
        return
    valid = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[valid], rgb[valid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0).astype(np.float64))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)


def write_task_results(log_dir, task_name, episode_records, step_limit):
    """Persist per-episode eval results for one task (GemBench-style).

    Writes, under {log_dir}/{task_name}/:
      * result.json       -- JSONL, one line per episode:
            {"episode_id", "instr", "success", "error", "nsteps"}
      * result_detail.txt -- human-readable summary: success rate, #samples,
            and every (episode_id, instr, success, nsteps) row.

    `episode_records` is the list of per-episode dicts collected during the
    rollout. Mirrors GemBench's result.json so that the same downstream
    summarizers can read RLBench eval output.
    """
    task_dir = os.path.join(log_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)

    # result.json (JSONL, overwrite each run so it matches this run exactly).
    result_json = os.path.join(task_dir, "result.json")
    with open(result_json, "w") as fp:
        for rec in episode_records:
            fp.write(json.dumps(rec) + "\n")

    n = len(episode_records)
    n_success = sum(1 for r in episode_records if r["success"] == 1.0)
    sr = (n_success / n * 100.0) if n else float("nan")

    detail_path = os.path.join(task_dir, "result_detail.txt")
    with open(detail_path, "w") as fp:
        fp.write(f"task = {task_name}\n")
        fp.write(f"step_limit = {step_limit}\n")
        fp.write(f"num_episodes = {n}\n")
        fp.write(f"num_success = {n_success}\n")
        sr_s = f"{sr:.2f}%" if n else "N/A"
        fp.write(f"success_rate = {sr_s}\n")
        fp.write("\n")
        fp.write(f"{'episode':>8s}{'success':>9s}{'nsteps':>8s}  {'instruction':<40s}{'error'}\n")
        fp.write("-" * 90 + "\n")
        for rec in episode_records:
            succ = "success" if rec["success"] == 1.0 else "fail"
            err = rec["error"] if rec["error"] is not None else ""
            fp.write(
                f"{rec['episode_id']:>8d}{succ:>9s}{rec['nsteps']:>8d}  "
                f"{str(rec['instr']):<40s}{err}\n"
            )

    return sr, n, n_success


@torch.no_grad()
def eval(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    step_limits=None,
    replay_ground_truth=False,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
    save_pointcloud=False,
    model_name="debug",
    visualize=False,
    visualize_root_dir="",
):
    agent.eval()

    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    # MoveArmThenGripper2 routes the 9-D action's trailing ignore_collisions
    # flag to the arm planner; the generic MoveArmThenGripper mis-feeds it to
    # the gripper (shape-(2,) -> InvalidActionError every step).
    action_mode = MoveArmThenGripper2(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None

        # create metric saving writer; embed the checkpoint index + run
        # timestamp (log_dir's own basename, per the eval/<model>/<ts> layout
        # in __main__) in the filename so the CSV stays identifiable once
        # copied out of its eval/<model>/<ts>/ directory.
        model_idx = get_model_index(model_name)
        model_tag = model_idx if model_idx is not None else model_name
        run_ts = os.path.basename(os.path.normpath(log_dir))
        csv_file = f"eval_results_{model_tag}_{run_ts}.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = [
                    "task", "success rate", "length",
                    "total_transitions", "num_episodes",
                ]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    current_task_id = -1

    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    step_limits = step_limits or {}

    scores = []
    for task_id in range(num_tasks):
        # Per-task step limit (RMBench-style): table value wins, else the
        # global --episode-length. Drive BOTH the rollout loop length and the
        # env's time_in_state / video-final-frame denominator off the same
        # per-task value (they were coupled on one episode_length before).
        task_episode_length = step_limits.get(tasks[task_id], episode_length)
        eval_env._episode_length = task_episode_length
        if verbose:
            print(
                f"[Evaluation] Task {tasks[task_id]} | step_limit = {task_episode_length}"
            )
        task_rewards = []
        language_goals=[]
        episode_ids=[]
        # Per-episode records for GemBench-style result.json (persisted after
        # the task finishes via write_task_results).
        episode_records = []
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            if save_pointcloud:
                task_name_for_pc = tasks[task_id]
                if not visualize:
                    pc_episode_dir = os.path.join(
                        log_dir, "pointclouds", task_name_for_pc, f"episode_{ep}",
                    )
                else:
                    pc_episode_dir = os.path.join(
                        visualize_root_dir, task_name_for_pc, f"episode_{ep}", "pointclouds",
                    )
                os.makedirs(pc_episode_dir, exist_ok=True)
            if not visualize:
                generator = rollout_generator.generator(
                    step_signal=step_signal,
                    env=eval_env,
                    agent=agent,
                    episode_length=task_episode_length,
                    timesteps=1,
                    eval=True,
                    eval_demo_seed=ep,
                    record_enabled=False,
                    replay_ground_truth=replay_ground_truth,
                )
            else:
                task_name = tasks[task_id]
                lang_goal = eval_env._lang_goal
                visualize_save_dir=os.path.join(visualize_root_dir,task_name,f"episode_{ep}")
                if not os.path.exists(visualize_save_dir):
                    os.makedirs(visualize_save_dir)
                generator = rollout_generator.generator_visualize(
                    step_signal=step_signal,
                    env=eval_env,
                    agent=agent,
                    episode_length=task_episode_length,
                    timesteps=1,
                    eval=True,
                    eval_demo_seed=ep,
                    record_enabled=True,
                    visualize_save_dir=visualize_save_dir,
                    visualize=True,
                )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
                    if save_pointcloud:
                        step_idx = len(episode_rollout) - 1
                        save_step_pointcloud(
                            replay_transition.observation, CAMERAS,
                            os.path.join(pc_episode_dir, f"step_{step_idx:03d}.ply"),
                        )

            except StopIteration as e:
                # Episode produced no transitions (e.g. planner failed at
                # reset). Still record it as a failed sample so the counts /
                # instructions stay aligned in result.json.
                episode_records.append({
                    "episode_id": ep,
                    "instr": eval_env._lang_goal,
                    "success": 0.0,
                    "error": "StopIteration",
                    "nsteps": len(episode_rollout),
                })
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            # Episode-end overlay viz: stitch this episode's per-step overlay
            # tri-views into grid_mvt1.png / grid_mvt2.png (rows=steps,
            # cols=views) under the episode's viz dir. Mirrors RMBench's
            # finalize_episode_viz; no-op when --visualize is off.
            if visualize:
                try:
                    agent.finalize_eval_viz()
                except Exception as e:
                    print(f"[RLBench eval viz] finalize_eval_viz failed: {e}")

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            language_goals.append(lang_goal)
            episode_ids.append(ep)
            # reward is 100 for success, 0 otherwise (same test as the video
            # naming logic below); store as GemBench 0/1 success.
            episode_records.append({
                "episode_id": ep,
                "instr": lang_goal,
                "success": 1.0 if reward > 99 else 0.0,
                "error": None,
                "nsteps": len(episode_rollout),
            })
            if verbose:
                print(
                    f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]

        # Persist per-episode results (GemBench-style result.json +
        # result_detail.txt) — records how many samples ran, each instruction,
        # per-episode success / nsteps / error. Skips silently when there is no
        # log_dir to write into (logging disabled).
        n_episodes = len(episode_records)
        record_sr = float("nan")
        if log_dir is not None:
            record_sr, n_episodes, _ = write_task_results(
                log_dir, task_name, episode_records, task_episode_length,
            )

        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = [
                    "task", "success rate", "length",
                    "total_transitions", "num_episodes",
                ]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name, "num_episodes": n_episodes}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                # Fallback: if the accumulator produced no return summary
                # (e.g. every episode StopIteration'd early), fill success rate
                # from the per-episode records so the CSV column isn't blank.
                if csv_results.get("success rate") is None and n_episodes:
                    csv_results["success rate"] = record_sr
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

        if save_video:
            video_image_folder = f"./tmp/{model_name}/{task_name}"
            palette_image_folder = f"./tmp/{model_name}/palette_folder"
            palette_image_path=os.path.join(palette_image_folder,"palette.png")

            record_fps = 25
            if not visualize:
                record_folder = os.path.join(log_dir, "videos")
            else:
                record_folder = os.path.join(visualize_root_dir,task_name,"videos")
            os.makedirs(record_folder, exist_ok=True)
            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0
            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    lang_goal = language_goals.pop(0)
                    lang_goal=lang_goal.replace(" ", "_")
                    ep_id = episode_ids[video_cnt]
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]
                    if task_rewards[video_cnt] > 99:
                        video_path = os.path.join(
                            record_folder,
                            f"episode_{ep_id}_{lang_goal}_success_{video_success_cnt}.mp4",
                        )
                        video_success_cnt += 1
                    else:
                        video_path = os.path.join(
                            record_folder, f"episode_{ep_id}_{lang_goal}_fail_{video_fail_cnt}.mp4"
                        )
                        video_fail_cnt += 1
                    video_cnt += 1
                    os.makedirs(video_image_folder, exist_ok=True)
                    os.makedirs(palette_image_folder, exist_ok=True)
                    for idx in range(len(video) - 10):
                        cv2.imwrite(
                            os.path.join(video_image_folder, f"{idx}.png"), video[idx]
                        )
                    images_path = os.path.join(video_image_folder, r"%d.png")
                    os.system(
                        "ffmpeg -i {} -vf palettegen {} -hide_banner -loglevel error".format(
                            images_path, palette_image_path
                        )
                    )
                    
                    os.system(
                        "ffmpeg -framerate {} -i {} -i {} -lavfi paletteuse {} -hide_banner -loglevel error".format(
                            record_fps, images_path, palette_image_path, video_path
                        )
                    )
                    print(f'video saved - {task_name}')
                    os.remove(palette_image_path)
                    shutil.rmtree(video_image_folder)

    eval_env.shutdown()

    if logging:
        # Rows above were appended in eval order (EVAL_TASKS list order, not
        # necessarily alphabetical); re-sort the whole file by task name (a-z)
        # once eval finishes so the CSV on disk is always alphabetically sorted.
        csv_path = os.path.join(log_dir, csv_file)
        with open(csv_path) as csv_fp:
            reader = csv.DictReader(csv_fp)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows.sort(key=lambda r: r.get("task") or "")
        with open(csv_path, "w") as csv_fp:
            csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(rows)

    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):

    model_paths = []
    assert args.model_name is not None
    model_paths.append(os.path.join(args.model_folder, args.model_name))
    tb = TensorboardManager(args.eval_log_dir)

    # Per-task step-limit table (falls back to --episode-length per task).
    step_limits = load_step_limits(args.step_limit_config)
    if step_limits:
        print(f"[Info] loaded per-task step limits from {args.step_limit_config}: "
              f"{step_limits}")
    for model_path in model_paths:
        tasks_to_eval = deepcopy(args.tasks)
        if tasks_to_eval[0] == "all":
            tasks_to_eval = deepcopy(RLBENCH_TASKS)
            print(f"evaluate on {len(tasks_to_eval)} tasks: ", tasks_to_eval)
        model_idx = get_model_index(model_path)
        if model_idx is None:
            model_idx = 0

  
        agent = load_agent(
            model_path=model_path,
            exp_cfg_path=args.exp_cfg_path,
            mvt_cfg_path=args.mvt_cfg_path,
            eval_log_dir=args.eval_log_dir,
            device=args.device,
            use_input_place_with_mean=args.use_input_place_with_mean,
            expect_temporal_memory=parse_bool_flag(
                args.temporal_memory, "--temporal_memory"),
            expect_spatial_memory=parse_bool_flag(
                args.spatial_memory, "--spatial_memory"),
        )

        # eval_log_dir already ends in eval/<model>/<date>; CSV/videos go
        # straight there (single checkpoint per run, no extra model subfolder).
        agent_eval_log_dir = args.eval_log_dir

        os.makedirs(agent_eval_log_dir, exist_ok=True)
        scores = eval(
            agent=agent,
            tasks=tasks_to_eval,
            eval_datafolder=args.eval_datafolder,
            start_episode=args.start_episode,
            eval_episodes=args.eval_episodes,
            episode_length=args.episode_length,
            step_limits=step_limits,
            replay_ground_truth=args.ground_truth,
            device=args.device,
            headless=args.headless,
            visualize=args.visualize,
            logging=True,
            log_dir=agent_eval_log_dir,
            verbose=True,
            save_video=args.save_video,
            save_pointcloud=args.save_pointcloud,
            model_name=args.model_name,
            visualize_root_dir=args.visualize_root_dir,
        )
        print(f"model {model_path}, scores {scores}")
        task_scores = {}
        for i in range(len(tasks_to_eval)):
            task_scores[tasks_to_eval[i]] = scores[i]

        print("save ", task_scores)
        # tb.update writes each score as a tensorboard scalar; a task that
        # produced no episode summaries yields the string "unknown", which would
        # abort the whole summary step (float("unknown") -> ValueError). Only
        # push numeric scores.
        numeric_task_scores = {
            k: v for k, v in task_scores.items() if isinstance(v, (int, float))
        }
        dropped = [k for k in task_scores if k not in numeric_task_scores]
        if dropped:
            print(f"[Evaluation] WARNING: no score for {dropped} (skipped in tb).")
        tb.update("eval", model_idx, numeric_task_scores)
        tb.writer.flush()

    tb.close()


if __name__ == "__main__":
    parser = get_eval_parser()
    # Memory ablation switches (train/eval alignment): must match the loaded
    # model's mvt_cfg.yaml or load_agent raises at startup. Set via eval.sh
    # TEMPORAL_MEMORY / SPATIAL_MEMORY.
    parser.add_argument("--temporal_memory", type=str, default="true",
                        help="true/false; stage-1 temporal memory (memory 1) the model is expected to have")
    parser.add_argument("--spatial_memory", type=str, default="true",
                        help="true/false; stage-2 spatial anchor memory (memory 2) the model is expected to have")
    # Per-task step limit table (RMBench-style). Values in this yaml override
    # --episode-length per task; a task missing (or null) falls back to it.
    # Pass "" to disable and use --episode-length for every task.
    parser.add_argument("--step-limit-config", type=str,
                        default=DEFAULT_STEP_LIMIT_CONFIG,
                        help="yaml mapping task_name -> max steps; overrides --episode-length per task")

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = "none"

    # Layout eval outputs as eval/<model>/<date>/ : group by checkpoint first,
    # then by run date (args.log_name is the timestamp passed from eval.sh).
    model_stem = os.path.basename(args.model_name).split(".")[0]
    args.eval_log_dir = os.path.join(
        args.model_folder, "eval", model_stem, args.log_name
    )

    os.makedirs(args.eval_log_dir, exist_ok=True)

    if not args.visualize_root_dir:
        args.visualize_root_dir = os.path.join(args.eval_log_dir, "visualize")

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)
