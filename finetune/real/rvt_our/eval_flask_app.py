"""
BridgeVLA++  Real-Robot Inference Server
===========================================
Flask server that:
  1. Loads the BridgeVLA++ model (PaliGemma + MVT head).
  2. Exposes a ``/predict`` HTTP endpoint for real-robot action inference.

Environment variables (optional)::

    PALIGEMMA_PATH         local HuggingFace snapshot of PaliGemma-3b-pt-224

Usage::

    python eval_flask_app.py          # starts on 0.0.0.0:5000
"""

import os
import re
import sys
import glob
import json
import time
import base64
import datetime
import traceback

import numpy as np
import torch
from flask import Flask, request, jsonify, Response
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Environment silencers ────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# ── Derive project root from file location ──────────────────────────────────
# <repo>/finetune/real/rvt_our/eval_flask_app.py  →  <repo>
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGEVLA_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))

# ── Pretrained model paths (PaliGemma) ──────────────────────────────────────
os.environ.setdefault(
    "PALIGEMMA_PATH",
    os.path.join(BRIDGEVLA_REPO_ROOT, "data", "bridgevla_ckpt", "paligemma-3b-pt-224"),
)

# ── Path setup ───────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so `rvt_our` is importable as a package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# BridgeVLA model code (MVT, renderer, augmentation, etc.)
BRIDGEVLA_FINETUNE_DIR = os.path.join(BRIDGEVLA_REPO_ROOT, "finetune")
POINT_RENDERER_DIR = os.path.join(
    BRIDGEVLA_FINETUNE_DIR, "bridgevla", "libs", "point-renderer"
)
YARR_LIB_DIR = os.path.join(
    BRIDGEVLA_FINETUNE_DIR, "bridgevla", "libs", "YARR"
)
for p in [BRIDGEVLA_FINETUNE_DIR, POINT_RENDERER_DIR, YARR_LIB_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── BridgeVLA imports ────────────────────────────────────────────────────────
import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.config as default_exp_cfg
from bridgevla.mvt.mvt import MVT
from bridgevla.utils.rvt_utils import load_agent as load_agent_state
from bridgevla.utils.memory_switches import assert_eval_memory_matches_model
from bridgevla.utils.viz_utils import _save_memory_panel, _save_mvt2_memory_grid

# Local inference-only agent (no training / yarr dependency)
from rvt_our.models.bridgevla_agent import RVTAgent
from real.utils.peract_utils import SCENE_BOUNDS_REAL, IMAGE_SIZE, CAMERAS_REAL
from rvt_our.botarm import TOOL_INDEX
from rvt_our import eval_summary


# 1. Model loading
def load_agent(
    model_path: str,
    exp_cfg_path: str = None,
    mvt_cfg_path: str = None,
    device: int = 0,
    load_pretrain: bool = False,
    pretrain_path: str = None,
    expect_temporal_memory: bool = True,
    expect_spatial_memory: bool = True,
) -> RVTAgent:
    """
    Instantiate MVT + RVTAgent, load checkpoint, set eval mode.

    Args:
        model_path:    Path to ``model_*.pth`` checkpoint.
        exp_cfg_path:  (optional) experiment config YAML override.
        mvt_cfg_path:  (optional) MVT config YAML override.
        device:        CUDA device ordinal.
        load_pretrain: Whether MVT should load a pre-trained VLM checkpoint.
        pretrain_path: Directory with ``model.safetensors.*`` shards.
        expect_temporal_memory / expect_spatial_memory:
            Memory switches this eval EXPECTS of the loaded model (mirror of
            real_config.yaml's memory.temporal_memory / memory.spatial_memory,
            set via MEMORY_TEMPORAL / MEMORY_SPATIAL below). Checked against
            the checkpoint's dumped mvt_cfg.yaml — a mismatch raises instead
            of silently evaluating the wrong memory ablation.

    Returns:
        Ready-to-use :class:`RVTAgent` in eval mode.
    """
    device_str = f"cuda:{device}"
    model_folder = os.path.dirname(model_path)

    # ---- experiment config ----
    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path is not None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    old_place_with_mean = exp_cfg.rvt.place_with_mean
    exp_cfg.rvt.place_with_mean = True
    exp_cfg.freeze()

    # ---- MVT config ----
    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path is not None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
    mvt_cfg.freeze()

    # Guard: eval-side memory switches must match the trained model's memory
    # config (mirrors RMBench / RLBench eval). Fail fast on mismatch.
    assert_eval_memory_matches_model(
        mvt_cfg, expect_temporal_memory, expect_spatial_memory,
        where="real eval_flask_app",
    )

    # For stage-two, restore the original place_with_mean
    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    # ---- build network ----
    rvt = MVT(
        renderer_device=device_str,
        load_pretrain=load_pretrain,
        pretrain_path=pretrain_path,
        **mvt_cfg,
    )

    # ---- build agent ----
    agent = RVTAgent(
        network=rvt.to(device_str),
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS_REAL,
        cameras=CAMERAS_REAL,
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    agent.build(training=False, device=device_str)
    load_agent_state(model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent


# 2. Request deserialisation
def deserialize_data(data: dict, device: str = "cuda:0") -> dict:
    """
    Convert JSON payload → observation dict for ``agent.act_real()``.

    Expected keys in *data*:
        language_goal  (str)
        rgb            (base64-encoded ndarray, shape (1, 3, H, W))
        pcd            (base64-encoded ndarray, shape (1, 3, H, W))

    ``low_dim_state`` is accepted but **ignored** — the new BridgeVLA++
    model does not use proprioception.
    """

    def _b64_to_ndarray(b64_dict):
        raw = base64.b64decode(b64_dict["data"])
        return np.frombuffer(
            raw, dtype=np.dtype(b64_dict["dtype"])
        ).reshape(b64_dict["shape"])

    language_goal = data["language_goal"]
    rgb = torch.from_numpy(_b64_to_ndarray(data["rgb"]).copy()).to(device)
    pcd = torch.from_numpy(_b64_to_ndarray(data["pcd"]).copy()).to(device)

    observation = {
        "language_goal": [[[language_goal]]],
        "3rd": {"rgb": rgb, "pcd": pcd},
    }
    return observation


# 3. Flask application
app = Flask(__name__)
model: RVTAgent = None
cameras_view = ["3rd"]

# ---- Eval-run logging ------------------------------------------------------
# Everything a run produces is written next to the checkpoint that produced
# it, so a log folder can never be attributed to the wrong model::
#
#   <dir holding the .pth>/eval/<model_stem>/     see VIEW_LOG_DIR below
#     summary_by_variation.txt           success rate per variation (all of them)
#     <variation>/                       free text: basic|height|distractor|…|
#       summary.txt  episodes.csv        or a custom name (test) — chosen by the
#                                        CLIENT (EVAL_VARIATION), sent on
#                                        /reset; tally is PER variation
#       ep<NNN>_<mmdd_HHMMSS>_<instruction_slug>_<result>/  result ∈
#         episode_meta.txt               success|fail|aborted
#                                        episode no., variation, checkpoint,
#                                        instruction, times, steps, result
#         step_0000/
#           meta.txt                     predicted pose / gripper for this step
#           views/                       everything this server renders
#           mvt1/{original_i,overlay_i,logits_i}.png
#           mvt2/...
#           memory_grid.png   mvt2_memory_grid.png
#         frames/                        written by the eval CLIENT's
#           frame_000000.jpg ...         MotionFrameRecorder (arm-motion video)
#           timestamps.txt    step_meta.txt
#
# e.g. <ckpt dir>/eval/model_1500/lighting/
#          ep007_0718_190656_swap-the-two-eggplants_success/step_0000/views/mvt1/
#
# The run folder is minted by POST /reset (which receives the instruction and
# the variation) and renamed by POST /finish (which appends the result suffix
# and rebuilds summary.txt / episodes.csv / summary_by_variation.txt).
# STEP_DIR_FMT is a layout contract with utils/motion_frame_recorder.py, which
# writes the frames/ half of each step folder (and exec_frames_to_video.py,
# which reads it back) — keep the three in sync; the run-folder name is a
# contract with eval_summary.py, which allocates the ep<NNN> number and scans
# the folders back into a success/fail tally.
STEP_DIR_FMT = "step_{:04d}"
VIEWS_SUBDIR = "views"
_RESULT_SUFFIXES = ("_success", "_fail", "_aborted")
_predict_step_counter = 0
# 1-based episode number of the current run, allocated on /reset by counting
# the run folders already in _run_root. It deliberately does NOT live in this
# process: the operator restarts this server between episodes, so an in-memory
# counter would reset to 1 every time. Disk is the tally. Numbering is PER
# VARIATION — each axis is its own campaign with its own success rate.
_episode_index = 0
# Variation sub-folder holding the current run, both set by /reset from its
# payload. ``_run_root`` is the directory episodes and their summary.txt live
# in: VIEW_LOG_DIR/<variation>. It stays None until the first /reset, and
# _run_dir() then falls back to VIEW_LOG_DIR so a client that never calls
# /reset (or an older one that sends no variation) still logs somewhere.
_variation = ""
_run_root = None


def _slugify(text: str, max_len: int = 48) -> str:
    """Language instruction → safe folder-name fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "task"


def _run_dir() -> str:
    """Absolute path of the current eval-run folder."""
    return os.path.join(_run_root or VIEW_LOG_DIR, _run_name)


def _step_dir(step_idx: int) -> str:
    """Absolute path of one step's folder inside the current run."""
    return os.path.join(_run_dir(), STEP_DIR_FMT.format(step_idx))

# ---- Pure model-inference latency stats (per episode) ----------------------
# Only the act_real() forward pass is timed (CUDA-synchronized); request
# deserialization, view-logging PNG writes, and network transfer are excluded.
_inference_latencies = []

# ---- Eager model loading (runs at import / startup) ----
print("Starting BridgeVLA++ inference server ...")
print(f"Robot tool coordinate index (eval client via botarm): {TOOL_INDEX}")
print("Loading model, please wait ...")

# >>>>>>>  Checkpoint configuration (edit these for your setup)  <<<<<<<<
# Loads the released checkpoint directory by default (model_<E>.pth +
# exp_cfg.yaml + mvt_cfg.yaml side by side). To evaluate a run you trained
# yourself, point REAL_EVAL_RUN_DIR at it — a training run directory already
# has the same layout.
RELEASE_CKPT_DIR = (os.environ.get("BRIDGEVLA_RELEASE_CKPT_DIR")
                    or os.path.join(BRIDGEVLA_REPO_ROOT, "data", "bridgevla_ckpt", "bridgevla_plus"))
BASE_PATH     = os.environ.get("REAL_EVAL_RUN_DIR", os.path.join(RELEASE_CKPT_DIR, "real"))
MODEL_NAME    = os.environ.get("REAL_EVAL_MODEL_NAME", "model_1500.pth")
# A release directory usually holds a single weight file, so fall back to the
# only model_*.pth present when MODEL_NAME does not match.
if not os.path.isfile(os.path.join(BASE_PATH, MODEL_NAME)):
    _cands = sorted(glob.glob(os.path.join(BASE_PATH, "model_*.pth")))
    if len(_cands) == 1:
        MODEL_NAME = os.path.basename(_cands[0])
        print(f"[eval] MODEL_NAME auto-resolved to {MODEL_NAME} (only weight in {BASE_PATH})")

DEVICE        = 0
# Set these two if you want to initialise MVT from a pre-trained VLM snapshot
LOAD_PRETRAIN = False
PRETRAIN_PATH = None
# ---- Memory ablation switches (eval side) ----------------------------------
# These mirror memory.temporal_memory / memory.spatial_memory in the training
# config and declare the memory this eval expects the model to have:
#   MEMORY_TEMPORAL  stage-1 (mvt1, global/coarse) temporal memory: the two
#                    neighbouring frames plus the stage-1 initial anchor frame.
#   MEMORY_SPATIAL   stage-2 (mvt2, local/fine) spatial memory: the local
#                    per-view initial anchor frame.
# Loading a checkpoint reads its mvt_cfg.yaml and aborts on any mismatch (say
# the model trained without temporal memory but True is requested here), so an
# ablation can never silently run against the wrong weights. Set both to False
# for the no-memory ablation; the default is full memory.
MEMORY_TEMPORAL = True
MEMORY_SPATIAL  = True

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

model_path   = os.path.abspath(os.path.join(BASE_PATH, MODEL_NAME))
exp_cfg_path = os.path.join(BASE_PATH, "exp_cfg.yaml")
mvt_cfg_path = os.path.join(BASE_PATH, "mvt_cfg.yaml")

# ---- Where this run's logs go ----
# Anchored to the checkpoint itself, not to the repo: the eval log root is
# always ``<dir of the .pth>/eval/<model_stem>/``, wherever that checkpoint
# happens to live (data_fast/<run>/, data/bridgevla_data/logs_real/train/<run>/,
# an external disk, …). So the logs for model_1100.pth and model_1500.pth of
# the same training run stay separate, and every log folder sits next to the
# weights that produced it.
# Set REAL_EVAL_LOG_ROOT to redirect elsewhere (read-only / slow checkpoint FS);
# it replaces the ``<dir of the .pth>/eval`` part, the /<model_stem>/ stays.
_model_stem   = os.path.splitext(os.path.basename(model_path))[0]  # "model_1100"
EVAL_LOG_ROOT = os.environ.get("REAL_EVAL_LOG_ROOT") or os.path.join(
    os.path.dirname(model_path), "eval")
VIEW_LOG_DIR  = os.path.abspath(os.path.join(EVAL_LOG_ROOT, _model_stem))

# Current eval-run folder name under VIEW_LOG_DIR. Every POST /reset mints a
# fresh one (timestamp + instruction slug); this startup fallback only catches
# /predict from a client that never called /reset.
_run_name = datetime.datetime.now().strftime("%m%d_%H%M%S") + "_noreset"

try:
    model = load_agent(
        model_path=model_path,
        exp_cfg_path=exp_cfg_path,
        mvt_cfg_path=mvt_cfg_path,
        device=DEVICE,
        load_pretrain=LOAD_PRETRAIN,
        pretrain_path=PRETRAIN_PATH,
        expect_temporal_memory=MEMORY_TEMPORAL,
        expect_spatial_memory=MEMORY_SPATIAL,
    )
    print(f"Model loaded on cuda:{DEVICE} — server ready.")
    print(f"Checkpoint:    {model_path}")
    print(f"Eval log root: {VIEW_LOG_DIR}")
    # The operator restarts this server between episodes, so say where the
    # campaign stands rather than looking like episode 1 every boot. Per
    # variation, since that is the unit episodes are numbered and scored in.
    try:
        _groups = eval_summary.scan_variations(VIEW_LOG_DIR)
        _done = {k: v for k, v in _groups.items() if v}
        if _done:
            print("So far:")
            for _name, _eps in _done.items():
                print(f"  {_name:<14} "
                      f"{eval_summary.one_line(eval_summary.tally(_eps))} "
                      f"| next ep{eval_summary.next_episode_index(os.path.join(VIEW_LOG_DIR, _name)):03d}")
        else:
            print(f"So far:        no episodes logged for this checkpoint yet.")
        print(f"Variations:    {', '.join(eval_summary.VARIATIONS)}"
              f"  (or any custom name, e.g. test)")
    except Exception:
        traceback.print_exc()
    print(f"Listening at http://0.0.0.0:5000/predict")
except Exception as e:
    print(f"Model loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# 4. Inference endpoint
@app.route("/predict", methods=["POST"])
def predict():
    global model, _predict_step_counter
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        observation = deserialize_data(data, device=f"cuda:{DEVICE}")
        _language_goal = observation["language_goal"][0][0][0]
        print(f"[predict] step {_predict_step_counter} | instruction: {_language_goal!r}")

        # Time ONLY the model forward pass. CUDA kernels launch asynchronously,
        # so synchronize before/after to measure true GPU wall time.
        torch.cuda.synchronize(DEVICE)
        _t_infer = time.perf_counter()
        with torch.no_grad():
            target_pos, target_quat, target_gripper, views_info = model.act_real(
                observation, cameras_view, return_views=True
            )
        torch.cuda.synchronize(DEVICE)
        infer_latency = time.perf_counter() - _t_infer
        _inference_latencies.append(infer_latency)
        print(f"[latency] step {_predict_step_counter}: model inference "
              f"{infer_latency * 1000:.1f} ms | "
              f"mean {np.mean(_inference_latencies) * 1000:.1f} ms over "
              f"{len(_inference_latencies)} steps this episode")

        # ---- save rendered three-views to disk ----
        # Memory snapshot is keyed separately from the per-stage view dicts
        # (it has no originals/overlays layout) — pop before the stage loop.
        mem_views = views_info.pop("memory", None) if views_info else None
        if views_info:
            # step_XXXX/views/ keeps rendered PNGs out of the client's
            # step_XXXX/frames/ — the two writers share the step folder but
            # never the same subfolder.
            step_dir  = _step_dir(_predict_step_counter)
            views_dir = os.path.join(step_dir, VIEWS_SUBDIR)
            os.makedirs(views_dir, exist_ok=True)
            for stage_name, stage_data in views_info.items():
                stage_dir = os.path.join(views_dir, stage_name)
                os.makedirs(stage_dir, exist_ok=True)
                for i, orig in enumerate(stage_data["originals"]):
                    Image.fromarray(orig).save(
                        os.path.join(stage_dir, f"original_{i}.png"))
                for i, ov in enumerate(stage_data["overlays"]):
                    Image.fromarray(ov).save(
                        os.path.join(stage_dir, f"overlay_{i}.png"))
                # Logits color scale shared across this stage's views, adaptive
                # rather than a hardcoded [-10, 40] (same convention as
                # viz_utils._save_stage and the SAM2Act eval server): the fixed
                # range was an arbitrary early guess and clips/flattens
                # whenever a run's logits live elsewhere. Absolute per-view
                # numbers are still printed in each panel's title.
                _l_list = stage_data.get("logits", [])
                _l_stack = np.stack(_l_list) if len(_l_list) else None
                l_vmin = float(_l_stack.min()) if _l_stack is not None else 0.0
                l_vmax = float(_l_stack.max()) if _l_stack is not None else 1.0
                if l_vmax - l_vmin < 1e-6:
                    l_vmax = l_vmin + 1e-6
                for i, p_l in enumerate(_l_list):
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    im = ax.imshow(p_l, cmap="jet", vmin=l_vmin, vmax=l_vmax,
                                   interpolation="nearest")
                    ax.set_title(f"pred view {i}\n"
                                 f"min={p_l.min():.3f}  max={p_l.max():.3f}")
                    ax.set_axis_off()
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    fig.tight_layout()
                    fig.savefig(os.path.join(stage_dir, f"logits_{i}.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)

            # ---- memory grids: what the bank held when THIS step ran ----
            # memory_grid.png       — mvt1: anchor + K history + current
            #                         (invalid slots blacked out)
            # mvt2_memory_grid.png  — anchor re-rendered under this step's
            #                         zoom vs the current zoomed render
            if mem_views is not None:
                _step_label = f"(step {_predict_step_counter})"
                _save_memory_panel(
                    mem_views["panel"],
                    os.path.join(views_dir, "memory_grid.png"),
                    view_names=mem_views["view_names"],
                    step_label=_step_label,
                    row_labels=mem_views.get("row_labels"),
                )
                if mem_views["mvt2_current"] is not None:
                    _save_mvt2_memory_grid(
                        anchor_img_one=mem_views["mvt2_anchor"],
                        current_img_one=mem_views["mvt2_current"],
                        anchor_valid=mem_views["anchor_valid"],
                        save_path=os.path.join(views_dir,
                                               "mvt2_memory_grid.png"),
                        view_names=mem_views["view_names"],
                        step_label=_step_label,
                    )

            meta_lines = [
                f"step: {_predict_step_counter}",
                f"tool_index: {TOOL_INDEX}",
                f"instruction: {observation['language_goal'][0][0][0]}",
                f"target_pos: {target_pos.tolist()}",
                f"target_quat: {target_quat.tolist()}",
                f"target_gripper (training 0=open,1=close): {target_gripper.tolist()[0]}",
                f"robot_cmd (1=open,0=close): {1 - int(target_gripper.tolist()[0])}",
            ]
            if mem_views is not None:
                meta_lines.append(
                    f"memory: anchor_valid={mem_views['anchor_valid']} "
                    f"n_hist={mem_views['n_hist']}/{mem_views['k_temporal']}"
                )
            with open(os.path.join(step_dir, "meta.txt"), "w") as f:
                f.write("\n".join(meta_lines) + "\n")
            print(f"[view_log] saved views to {views_dir}")
        _predict_step_counter += 1

        result = {
            "target_pos":        target_pos.tolist(),
            "target_quat":       target_quat.tolist(),
            "target_gripper":    target_gripper.tolist(),
            # pure model forward time in seconds (excl. network / IO)
            "inference_latency": infer_latency,
        }
        print("result:", result)
        return Response(json.dumps(result), mimetype="application/json")

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "details": traceback.format_exc(),
        }), 500


# 4b. Episode-reset endpoint
@app.route("/reset", methods=["POST"])
def reset():
    """Clear the per-episode episodic-memory bank and open a fresh log dir.

    The server is long-lived; a single client run == a single episode. The
    client POSTs /reset once at startup so the next /predict re-anchors fresh
    (anchor = the first frame of the new episode) instead of carrying the
    previous episode's anchor / history tokens. No-op when the loaded
    checkpoint has memory disabled.

    The JSON body ``{"instruction": <language goal>, "variation": <axis>}``
    names the run folder. Each /reset mints a new ``<ckpt dir>/eval/
    <model_stem>/<variation>/ep<NNN>_<mmdd_HHMMSS>_<instruction_slug>/``
    directory so view PNGs and client-side motion frames for this episode
    share one place; its absolute path is returned as ``session_dir``.
    ``episode_meta.txt`` is seeded here — the server owns the run folder, so
    it also owns the run's metadata — and POST /finish later appends the
    result to both the file and the folder name.

    ``variation`` is the generalization axis under test — free text (a standard
    axis in ``eval_summary.VARIATIONS`` or a custom name like ``test``). It is
    validated rather than trusted, but only against names that would break the
    folder path (empty, a ``/`` or ``\\`` separator, ``.`` / ``..``); any usable
    name is kept as its own bucket. Omitting it is allowed and keeps the pre-split
    layout (runs directly under the model dir), so an older client still works.

    ``ep<NNN>`` is the episode counter **within that variation**. It is
    allocated from the folders already there (max + 1), not from process state,
    so it keeps counting across the server restarts the operator does between
    episodes. Returned as ``episode_index``.

    Single-operator real-robot use issues no concurrent requests, so the
    global model + step counter need no lock; add one if you ever drive
    multiple arms against the same server.
    """
    global model, _predict_step_counter, _run_name, _episode_index
    global _variation, _run_root
    try:
        model.reset()
        _predict_step_counter = 0  # restart per-episode view-log numbering
        if _inference_latencies:
            lat_ms = np.array(_inference_latencies) * 1000.0
            print(f"[latency] episode summary over {len(lat_ms)} steps: "
                  f"mean {lat_ms.mean():.1f} ms | min {lat_ms.min():.1f} ms | "
                  f"max {lat_ms.max():.1f} ms")
        _inference_latencies.clear()  # per-episode latency stats
        data = request.get_json(silent=True) or {}
        instruction = str(data.get("instruction") or "").strip()
        # Variation is free text; only reject a name that would break or escape
        # the folder path (validate_variation), not an unfamiliar one. Empty
        # means "no variation" (pre-split layout), so it skips validation.
        _variation = str(data.get("variation") or "").strip()
        if _variation:
            eval_summary.validate_variation(_variation)
        _run_root = os.path.join(VIEW_LOG_DIR, _variation) if _variation \
            else VIEW_LOG_DIR
        os.makedirs(_run_root, exist_ok=True)
        _episode_index = eval_summary.next_episode_index(_run_root)
        _run_name = eval_summary.format_run_name(
            _episode_index,
            datetime.datetime.now().strftime("%m%d_%H%M%S"),
            _slugify(instruction) if instruction else "",
        )
        run_dir = _run_dir()
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "episode_meta.txt"), "w") as f:
            f.write(f"episode: {_episode_index}\n"
                    f"variation: {_variation}\n"
                    f"checkpoint: {model_path}\n"
                    f"memory_temporal: {MEMORY_TEMPORAL}\n"
                    f"memory_spatial: {MEMORY_SPATIAL}\n"
                    f"instruction: {instruction}\n"
                    f"start_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Echo the bank's post-reset state rather than just claiming it was
        # cleared: restarting only the client (not this server) between
        # episodes is only safe if this really is a fresh bank, and this line
        # is the per-episode proof. Expect first_frame=True slots=0 every time
        # — anything else means episode N-1 leaked into episode N.
        try:
            _bank = model.memory_bank
            print(f"[reset] memory bank cleared: first_frame="
                  f"{_bank.first_frame()} slots={len(_bank.ordered_slots())} "
                  f"(expect True / 0); step counter reset.")
        except Exception:
            print("[reset] memory bank cleared; step counter reset.")
        print(f"[reset] EPISODE {_episode_index} "
              f"[{_variation or 'no variation'}] — eval log dir: {run_dir}")
        return Response(
            json.dumps({"status": "ok", "session_dir": run_dir,
                        "episode_index": _episode_index,
                        "variation": _variation}),
            mimetype="application/json",
        )
    except ValueError as e:          # unknown variation — the client's bug
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 4c. Episode-finish endpoint
@app.route("/finish", methods=["POST"])
def finish():
    """Close the current eval run: record its result, rename it, re-tally.

    The client POSTs ``{"result": "success" | "fail" | "aborted"}`` once at
    shutdown. The result is appended to ``episode_meta.txt`` and to the run
    folder name (``…_success`` / ``…_fail`` / ``…_aborted``); the renamed
    absolute path is returned as ``session_dir``. A second POST for the same
    run is a no-op that reports the already-renamed path.

    Closing an episode is also what updates the campaign tally: every finish
    rescans the current variation's folder and rewrites its ``summary.txt`` +
    ``episodes.csv``, then refreshes the model-level
    ``summary_by_variation.txt`` roll-up. Rebuilding from scratch (rather than
    appending a row) means the files can never disagree with what is on disk —
    delete a botched episode's folder and the next finish drops it from the
    counts. The tally returned as ``tally`` is for THIS variation (the number
    the operator is currently driving up); ``overall`` spans all variations.
    """
    global _run_name
    data = request.get_json(silent=True) or {}
    result = str(data.get("result") or "").strip().lower()
    if result not in ("success", "fail", "aborted"):
        return jsonify({"error": f"result must be one of success|fail|aborted,"
                                 f" got {result!r}"}), 400
    try:
        old_dir = _run_dir()
        if _run_name.endswith(_RESULT_SUFFIXES):
            return Response(
                json.dumps({"status": "already_finished",
                            "session_dir": old_dir}),
                mimetype="application/json",
            )
        if not os.path.isdir(old_dir):
            return Response(
                json.dumps({"status": "no_session", "session_dir": None}),
                mimetype="application/json",
            )
        with open(os.path.join(old_dir, "episode_meta.txt"), "a") as f:
            f.write(f"steps: {_predict_step_counter}\n"
                    f"result: {result}\n"
                    f"end_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _run_name += "_" + result
        new_dir = _run_dir()
        os.rename(old_dir, new_dir)
        print(f"[finish] EPISODE {_episode_index} "
              f"[{_variation or 'no variation'}] {result.upper()} → {new_dir}")
        # Never let a tally problem fail the episode that was just recorded:
        # the rename above is the part that must not be lost.
        stats = overall = None
        try:
            root = _run_root or VIEW_LOG_DIR
            episodes = eval_summary.scan_episodes(root)
            stats = eval_summary.write_summary(root, episodes,
                                               checkpoint=model_path)
            print(eval_summary.format_summary(episodes, root=root,
                                              checkpoint=model_path))
            if _variation:
                overall = eval_summary.write_rollup(VIEW_LOG_DIR,
                                                    checkpoint=model_path)
                print(eval_summary.format_rollup(VIEW_LOG_DIR,
                                                 checkpoint=model_path))
        except Exception:
            print("[finish] WARNING: could not rebuild the episode summary:")
            traceback.print_exc()
        return Response(
            json.dumps({"status": "ok", "result": result,
                        "session_dir": new_dir,
                        "episode_index": _episode_index,
                        "variation": _variation,
                        "tally": stats, "overall": overall}),
            mimetype="application/json",
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 4d. Campaign-summary endpoint
@app.route("/summary", methods=["GET"])
def summary():
    """Success rate per variation for this checkpoint, without ending an episode.

    ``curl -s localhost:5000/summary?format=text`` mid-campaign prints the same
    roll-up table /finish does; ``?variation=lighting`` narrows to one axis's
    episode list. Read-only apart from refreshing the summary files.
    """
    try:
        want = request.args.get("variation")
        as_text = request.args.get("format") == "text"
        if want:
            eval_summary.validate_variation(want)
            root = os.path.join(VIEW_LOG_DIR, want)
            episodes = eval_summary.scan_episodes(root)
            stats = eval_summary.write_summary(root, episodes,
                                               checkpoint=model_path)
            text = eval_summary.format_summary(episodes, root=root,
                                               checkpoint=model_path)
        else:
            groups = eval_summary.scan_variations(VIEW_LOG_DIR)
            stats = eval_summary.write_rollup(VIEW_LOG_DIR, groups,
                                              checkpoint=model_path)
            episodes = [e for eps in groups.values() for e in eps]
            text = eval_summary.format_rollup(VIEW_LOG_DIR, groups,
                                              checkpoint=model_path)
        if as_text:
            return Response(text + "\n", mimetype="text/plain")
        return Response(
            json.dumps({"status": "ok", "log_dir": VIEW_LOG_DIR,
                        "checkpoint": model_path,
                        "variation": want or None, "tally": stats,
                        "episodes": episodes}),
            mimetype="application/json",
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# 5. Entry point
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )