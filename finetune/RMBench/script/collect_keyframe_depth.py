"""
Collect depth data ONLY at keyframe indices.

Strategy:
- Full physics replay (same seed + _traj_data) to ensure correct scene state
- Skip rendering on non-keyframe frames (the main bottleneck)
- Only call cameras.update_picture() + get_obs() at keyframe frame indices
- Output: one HDF5 per episode containing only keyframe observations (with depth)

This is ~30-40x faster than full collection since rendering is skipped for >95% frames.
"""
import sys
sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
import numpy as np
import yaml
import importlib
import json
import os
import h5py
import pickle
from argparse import ArgumentParser
from copy import deepcopy

from envs import *
from envs.utils import save_pkl
from envs.utils.pkl2hdf5 import create_hdf5_from_dict, parse_dict_structure, append_data_to_structure, load_pkl_file as load_pkl

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

KEYFRAMES_DIR = os.path.join(DATA_PATH, "keyframes")

# Output root for the (color-corrected) keyframe dataset — the tree that
# RMBench_vla training consumes and that the release ships. Kept separate from
# the legacy "data_self" so the old R<->B-swapped data is preserved for
# comparison. See pkl2hdf5.images_encoding for the RGB->BGR fix that makes the
# stored JPEGs standard (PIL-decodable) RGB.
KEYFRAME_OUTPUT_ROOT = "keyframe_data"


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_keyframes(task_name):
    """Load keyframe indices from the precomputed JSON."""
    kf_path = os.path.join(KEYFRAMES_DIR, f"{task_name}.json")
    if not os.path.exists(kf_path):
        raise FileNotFoundError(f"Keyframe file not found: {kf_path}")
    with open(kf_path, "r") as f:
        data = json.load(f)
    return data


def monkey_patch_take_picture(task_env, keyframe_set):
    """
    Replace _take_picture so that:
    - It always increments FRAME_IDX (to keep tracking correct)
    - But only renders and saves data when FRAME_IDX is in keyframe_set
    """
    original_get_obs = task_env.get_obs

    def patched_take_picture(self_ref=task_env):
        if not self_ref.save_data:
            return

        self_ref.language_annotation_cache += 1

        if self_ref.FRAME_IDX == 0:
            self_ref.folder_path = {"cache": f"{self_ref.save_dir}/.cache/episode{self_ref.ep_num}/"}
            for directory in self_ref.folder_path.values():
                if os.path.exists(directory):
                    file_list = os.listdir(directory)
                    for file in file_list:
                        os.remove(directory + file)

        if self_ref.FRAME_IDX in keyframe_set:
            # Only render at keyframe frames
            pkl_dic = original_get_obs()
            save_pkl(self_ref.folder_path["cache"] + f"{self_ref.FRAME_IDX}.pkl", pkl_dic)

        self_ref.FRAME_IDX += 1

    task_env._take_picture = patched_take_picture


def merge_keyframe_pkls_to_hdf5(cache_path, hdf5_path, keyframe_indices):
    """Merge only keyframe pkl files into a single HDF5."""
    pkl_files = []
    for idx in sorted(keyframe_indices):
        pkl_path = os.path.join(cache_path, f"{idx}.pkl")
        if os.path.exists(pkl_path):
            pkl_files.append(pkl_path)

    if not pkl_files:
        raise FileNotFoundError(f"No keyframe pkl files found in {cache_path}")

    # Build structure from first pkl
    first_data = load_pkl(pkl_files[0])
    data_list = parse_dict_structure(first_data)

    for pkl_path in pkl_files:
        pkl_data = load_pkl(pkl_path)
        append_data_to_structure(data_list, pkl_data)

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)
        f.create_dataset("keyframe_indices", data=np.array(sorted(keyframe_indices)))

    print(f"  Saved {len(pkl_files)} keyframes to {hdf5_path}")


def main(task_name=None):
    task_config = "demo_clean_depth"
    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        # See the comment on the same function in collect_data.py: strip the assets prefix, anchor to ASSETS_PATH, return an absolute path.
        rel = os.path.normpath(robot_file).split(os.sep)
        if rel and rel[0] == "assets":
            rel = rel[1:]
        return os.path.join(ASSETS_PATH, *rel)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    # Use same save_path structure but under a keyframe-specific dir
    args["save_path"] = os.path.join(DATA_PATH, KEYFRAME_OUTPUT_ROOT, str(args["task_name"]), "keyframe_depth")

    keyframes_data = load_keyframes(task_name)

    # Load seeds
    seed_path = os.path.join(args["save_path"], "seed.txt")
    src_seed_path = os.path.join(DATA_PATH, "data", task_name, "demo_clean", "seed.txt")
    os.makedirs(args["save_path"], exist_ok=True)

    # Copy seed and traj_data from original
    if not os.path.exists(seed_path):
        import shutil
        shutil.copy(src_seed_path, seed_path)

    src_traj = os.path.join(DATA_PATH, "data", task_name, "demo_clean", "_traj_data")
    dst_traj = os.path.join(args["save_path"], "_traj_data")
    if not os.path.exists(dst_traj):
        import shutil
        shutil.copytree(src_traj, dst_traj)

    with open(seed_path, "r") as file:
        seed_list = [int(i) for i in file.read().split()]

    # Collect data
    print(f"\033[93m[Keyframe Depth Collection] Task: {task_name}\033[0m")
    print(f"  Episodes: {args['episode_num']}, using keyframe-only rendering")

    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = True

    clear_cache_freq = args["clear_cache_freq"]
    output_dir = os.path.join(args["save_path"], "data")
    os.makedirs(output_dir, exist_ok=True)

    # Resume support: gap-fill. Skip episodes whose HDF5 already exists so a
    # re-run only (re)collects MISSING episodes (e.g. ones that previously
    # failed on physics non-determinism), instead of stopping at the first gap
    # and re-collecting everything after it.
    max_retries = 3
    failed_episodes = []

    for episode_idx in range(args["episode_num"]):
        if os.path.exists(os.path.join(output_dir, f"episode{episode_idx}.hdf5")):
            continue
        episode_key = f"episode{episode_idx}"
        if episode_key not in keyframes_data:
            print(f"  WARNING: {episode_key} not in keyframes JSON, skipping")
            continue

        kf_info = keyframes_data[episode_key]
        keyframe_indices = set(kf["frame_idx"] for kf in kf_info["keyframes"])

        print(f"  Episode {episode_idx}: {len(keyframe_indices)} keyframes / {kf_info['n_frames']} total frames")

        expected_n_frames = kf_info["n_frames"]
        success = False
        for attempt in range(max_retries):
            try:
                # Monkey-patch to only render at keyframes
                monkey_patch_take_picture(task, keyframe_indices)

                task.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

                traj_data = task.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                task.set_path_lst(args)

                task.play_once()

                # Verify FRAME_IDX matches expected total frames (detect diverged execution path)
                actual_frames = task.FRAME_IDX
                if actual_frames != expected_n_frames:
                    raise RuntimeError(
                        f"Frame count mismatch: got {actual_frames}, expected {expected_n_frames}. "
                        f"Execution path diverged due to physics non-determinism."
                    )

                # Verify all keyframe pkl files exist
                cache_path = task.folder_path["cache"]
                missing_kf = [idx for idx in sorted(keyframe_indices) if not os.path.exists(os.path.join(cache_path, f"{idx}.pkl"))]
                if missing_kf:
                    raise RuntimeError(f"Missing keyframe pkl files: {missing_kf}")

                # Merge keyframe pkls to HDF5
                hdf5_path = os.path.join(output_dir, f"episode{episode_idx}.hdf5")
                merge_keyframe_pkls_to_hdf5(cache_path, hdf5_path, keyframe_indices)

                # Post-write validation: compare joint_action with original data
                orig_hdf5_path = os.path.join(DATA_PATH, "data", task_name, "demo_clean", "data", f"episode{episode_idx}.hdf5")
                if os.path.exists(orig_hdf5_path):
                    import h5py as h5val
                    with h5val.File(hdf5_path, "r") as kf_f, h5val.File(orig_hdf5_path, "r") as orig_f:
                        sorted_kf = sorted(keyframe_indices)
                        for i, fi in enumerate(sorted_kf):
                            kf_vec = np.array(kf_f["joint_action/vector"][i])
                            orig_vec = np.array(orig_f["joint_action/vector"][fi])
                            max_diff = np.max(np.abs(kf_vec - orig_vec))
                            if max_diff > 1e-4:
                                # Data is corrupted - delete and raise
                                os.remove(hdf5_path)
                                raise RuntimeError(
                                    f"joint_action mismatch at keyframe {fi} (idx {i}): "
                                    f"max_diff={max_diff:.6f} > 1e-4. Data deleted."
                                )

                success = True
                break
            except Exception as e:
                print(f"\033[93m  [RETRY] Episode {episode_idx} failed (attempt {attempt+1}/{max_retries}): {e}\033[0m")
                # Clean up failed attempt
                try:
                    task.close_env(clear_cache=True)
                except:
                    pass
                failed_hdf5 = os.path.join(output_dir, f"episode{episode_idx}.hdf5")
                if os.path.exists(failed_hdf5):
                    os.remove(failed_hdf5)
                # Also clean cache from failed attempt
                try:
                    import shutil
                    cache_path = task.folder_path.get("cache", "")
                    if cache_path and os.path.exists(cache_path):
                        shutil.rmtree(cache_path)
                except:
                    pass
                import time
                time.sleep(0.5)

        if success:
            # Cleanup cache
            try:
                task.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            except:
                pass
            import shutil
            cache_path = task.folder_path.get("cache", "")
            if cache_path and os.path.exists(cache_path):
                shutil.rmtree(cache_path)
        else:
            print(f"\033[91m  [FAILED] Episode {episode_idx} failed after {max_retries} attempts. Skipping.\033[0m")
            failed_episodes.append(episode_idx)

    if failed_episodes:
        failed_path = os.path.join(args["save_path"], "failed_episodes.json")
        with open(failed_path, "w") as f:
            json.dump({"failed": failed_episodes, "total": args["episode_num"],
                        "success_rate": 1 - len(failed_episodes)/args["episode_num"]}, f, indent=2)
        print(f"\033[93m[Summary] {len(failed_episodes)}/{args['episode_num']} episodes failed: {failed_episodes}\033[0m")

    print(f"\033[92m[Done] {task_name}: keyframe depth collection complete.\033[0m")


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser = parser.parse_args()

    main(task_name=parser.task_name)
