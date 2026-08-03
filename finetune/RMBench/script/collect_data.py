import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


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


def main(task_name=None, task_config=None):

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
        # file_path historically read "./assets/embodiments/..." relative to finetune/RMBench/, resolved
        # through the data/assets symlink. That symlink is gone and assets moved to the data root, so the
        # assets prefix is stripped here and the rest anchored to ASSETS_PATH, returning an absolute path
        # (accepting "./assets/...", "assets/..." and "embodiments/...").
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
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
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

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    # By convention save_path in the config is written "./data", which used to resolve to the real data
    # root through the finetune/RMBench/data symlink; that symlink is gone, so relative paths are anchored to DATA_PATH (absolute paths are respected as-is).
    _save_root = args["save_path"] if os.path.isabs(args["save_path"]) else DATA_PATH
    args["save_path"] = os.path.join(_save_root, str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]
        max_retries = args.get("max_retries", 3) if args.get("use_seed", False) else 1

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        language_annotation_json_path = os.path.join(args["save_path"], 'language_annotation.json')
        failed_episodes = []

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            success = False
            for attempt in range(max_retries):
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

                traj_data = TASK_ENV.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")

                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)

                if not os.path.exists(language_annotation_json_path):
                    with open(language_annotation_json_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)
                with open(language_annotation_json_path, "r", encoding="utf-8") as file:
                    language_annotation = json.load(file)
                language_annotation[f'episode_{episode_idx}'] = TASK_ENV.language_annotation
                with open(language_annotation_json_path, "w", encoding="utf-8") as file:
                    json.dump(language_annotation, file, ensure_ascii=False, indent=4)

                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                TASK_ENV.merge_pkl_to_hdf5_video()
                TASK_ENV.remove_data_cache()

                if TASK_ENV.check_success():
                    success = True
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"\033[93m[RETRY] Episode {episode_idx} failed (attempt {attempt+1}/{max_retries}), retrying...\033[0m")
                        # Remove failed HDF5 and video before retry
                        failed_hdf5 = os.path.join(args["save_path"], 'data', f'episode{episode_idx}.hdf5')
                        failed_video = os.path.join(args["save_path"], 'video', f'episode{episode_idx}.mp4')
                        if os.path.exists(failed_hdf5):
                            os.remove(failed_hdf5)
                        if os.path.exists(failed_video):
                            os.remove(failed_video)

            if not success:
                if args.get("use_seed", False):
                    print(f"\033[91m[FAILED] Episode {episode_idx} failed after {max_retries} attempts. Skipping (data removed).\033[0m")
                    failed_hdf5 = os.path.join(args["save_path"], 'data', f'episode{episode_idx}.hdf5')
                    failed_video = os.path.join(args["save_path"], 'video', f'episode{episode_idx}.mp4')
                    if os.path.exists(failed_hdf5):
                        os.remove(failed_hdf5)
                    if os.path.exists(failed_video):
                        os.remove(failed_video)
                    failed_episodes.append(episode_idx)
                else:
                    assert False, "Collect Error"

        if failed_episodes:
            failed_path = os.path.join(args["save_path"], "failed_episodes.json")
            with open(failed_path, "w") as f:
                json.dump({"failed": failed_episodes, "total": args["episode_num"], "success_rate": 1 - len(failed_episodes)/args["episode_num"]}, f, indent=2)
            print(f"\033[93m[Summary] {len(failed_episodes)}/{args['episode_num']} episodes failed: {failed_episodes}\033[0m")

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
