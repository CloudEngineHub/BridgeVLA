# global configs
import os

# ROOT_PATH = the root of the RMBench code tree (the parent of envs/), always finetune/RMBench/.
# task_config / script / description belong to the code tree and follow ROOT_PATH.
ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/")]
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/") + 1]

# data/ and assets/ do not live in the code tree; they sit together under the data root data/bridgevla_data/RMBench/.
# The RMBENCH_ASSETS_PATH / RMBENCH_DATA_PATH environment variables take priority (the eval/collect scripts export them);
# when unset (e.g. launching python directly) they fall back to paths derived inside the repository:
#   <repo>/finetune/RMBench/envs/_GLOBAL_CONFIGS.py -> <repo>/data/bridgevla_data/RMBench/{assets,data}
_REPO_ROOT = os.path.abspath(os.path.join(ROOT_PATH, os.pardir, os.pardir))  # .../<repo>
_RMBENCH_DATA_ROOT = os.path.join(_REPO_ROOT, "data", "bridgevla_data", "RMBench")

ASSETS_PATH = os.environ.get("RMBENCH_ASSETS_PATH") or os.path.join(_RMBENCH_DATA_ROOT, "assets")
DATA_PATH = os.environ.get("RMBENCH_DATA_PATH") or os.path.join(_RMBENCH_DATA_ROOT, "data")
ASSETS_PATH = os.path.join(ASSETS_PATH, "")  # ensure a trailing "/", matching the old behaviour
DATA_PATH = os.path.join(DATA_PATH, "")

EMBODIMENTS_PATH = os.path.join(ASSETS_PATH, "embodiments/")
TEXTURES_PATH = os.path.join(ASSETS_PATH, "background_texture/")
CONFIGS_PATH = os.path.join(ROOT_PATH, "task_config/")
SCRIPT_PATH = os.path.join(ROOT_PATH, "script/")
DESCRIPTION_PATH = os.path.join(ROOT_PATH, "description/")

# Euler angles in world coordinates
# t3d.euler.quat2euler(quat) returns (theta_x, theta_y, theta_z)
# theta_y controls the pitch, and theta_z controls rotation around the axis perpendicular to the tabletop plane
GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}

WORLD_DIRECTION_DIC = {
    "left": [0, -0.707, 0, 0.707],  # -z  -y  -x
    "front": [0.5, -0.5, 0.5, 0.5],  # y   z   -x
    "right": [0.707, 0, 0.707, 0],  # z   y   -x
    "top_down": [0, 0.707, -0.707, 0],  # -x  -y  -z
}

ROTATE_NUM = 10
