"""
MemoryBench actioner. Hosts a BridgeVLA agent in-process and answers /predict
calls from client.py.

Almost identical to finetune/GemBench/actioner.py: same agent class, same
load_agent flow, same observation -> tensor conversion. The only delta is
peract_utils_memorybench (different IMAGE_SIZE / SCENE_BOUNDS handle if a
fork wants to diverge later) instead of peract_utils_gembench.

Episodic memory contract:
  step_id == 0 -> agent.reset() (clears anchor + history bank).
  This works because GemBench's client/server protocol has no /reset hook,
  and step_id is monotonic per-episode. Same convention as GemBench.
"""
import os

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import bridgevla.config as default_exp_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as default_mvt_cfg
from bridgevla.mvt.mvt import MVT
from bridgevla.utils.rvt_utils import load_agent as load_agent_state
from bridgevla.utils.memory_switches import assert_eval_memory_matches_model

from utils.peract_utils_memorybench import CAMERAS, SCENE_BOUNDS  # noqa: F401


class MyActioner(object):
    def __init__(self, base_path, model_epoch=40,
                 expect_temporal_memory=True, expect_spatial_memory=True):
        '''expect_temporal_memory / expect_spatial_memory: the eval-side memory
        ablation switches declared by run_server.sh (TEMPORAL_MEMORY /
        SPATIAL_MEMORY). load_agent raises if they disagree with the loaded
        checkpoint's mvt_cfg.yaml (train/eval alignment guard). The validated
        values are re-exposed via the server /memory_config route so the
        client can verify its own switches against the server.'''
        model_path = os.path.join(base_path, f"model_{model_epoch}.pth")
        print("[MemoryBench] model path:", model_path)
        exp_cfg_path = os.path.join(base_path, "exp_cfg.yaml")
        mvt_cfg_path = os.path.join(base_path, "mvt_cfg.yaml")
        self.expect_temporal_memory = bool(expect_temporal_memory)
        self.expect_spatial_memory = bool(expect_spatial_memory)
        self.agent = load_agent(
            model_path=model_path,
            exp_cfg_path=exp_cfg_path,
            mvt_cfg_path=mvt_cfg_path,
            device=0,
            use_input_place_with_mean=False,
            expect_temporal_memory=self.expect_temporal_memory,
            expect_spatial_memory=self.expect_spatial_memory,
        )

    def memory_config(self):
        '''Server /memory_config route payload: the eval-side memory switches
        this server was launched with (already validated against the loaded
        model). The client compares its own switches against these and aborts
        on mismatch (client/server alignment guard).'''
        return {
            "temporal_memory": self.expect_temporal_memory,
            "spatial_memory": self.expect_spatial_memory,
        }

    def predict(self, taskvar, episode_id, step_id, instruction, obs_state_dict,
                visualize=False, visualize_episode_dir=""):
        # Episode boundary: clear MemoryBank so the previous episode's anchor /
        # history doesn't leak. Required when memory.enabled=True; harmless
        # otherwise.
        if step_id == 0:
            self.agent.reset()

        for idx, cam in enumerate(self.agent.cameras):
            obs_state_dict[f"{cam}_rgb"] = np.transpose(obs_state_dict["rgb"][idx], [2, 0, 1])[None]
            obs_state_dict[f"{cam}_point_cloud"] = np.transpose(obs_state_dict["pc"][idx], [2, 0, 1])[None]

        for k in ("rgb", "pc", "arm_links_info", "depth", "gripper"):
            obs_state_dict.pop(k, None)

        for k, v in list(obs_state_dict.items()):
            if isinstance(v, np.ndarray):
                obs_state_dict[k] = torch.from_numpy(v).to(self.agent._device)
            elif isinstance(v, list):
                obs_state_dict[k] = torch.tensor(v).to(self.agent._device)
            elif isinstance(v, torch.Tensor):
                obs_state_dict[k] = v.to(self.agent._device)
            obs_state_dict[k] = obs_state_dict[k].unsqueeze(0)
        obs_state_dict["language_goal"] = [[[instruction]]]

        # return_gembench_action=True -> 8-D [wpt(3), quat(4), grip(1)]. The
        # collision slot is dropped unconditionally, matching GemBench. This is
        # the fixed convention for MemoryBench: the env is built with
        # EndEffectorPoseViaPlanning(collision_checking=False) and the stock
        # MoveArmThenGripper.action() takes no per-call ignore_collisions, so
        # the planner runs with ignore_collisions=True on every step regardless.
        # Previously this asked for the 9-D action and client.py read slot 8 and
        # then forced it to 1 — identical behaviour, one extra hop and a debug
        # print per step. Old checkpoints (trained when the collision head was
        # supervised) evaluate exactly the same here.
        #
        # torch.no_grad: RVTAgent.act (unlike act_rmbench) is NOT decorated
        # with @torch.no_grad, so without this the two-stage 3B forward
        # builds a full autograd graph — activations balloon the ~8 GiB
        # loaded model to 35+ GiB and OOM on a shared GPU.
        with torch.no_grad():
            action = self.agent.act(
                step=step_id,
                observation=obs_state_dict,
                visualize=visualize,
                visualize_save_dir=visualize_episode_dir,
                return_gembench_action=True,
            )
        return action

    def finalize_episode(self, **kwargs):
        """Episode-end hook (called via the server /finalize route). Stitch the
        just-finished episode's per-step overlay tri-views into per-stage grids
        (grid_mvt1.png / grid_mvt2.png). No-op when overlay viz wasn't active."""
        try:
            return self.agent.finalize_eval_viz()
        except Exception as e:
            print(f"[MemoryBench eval viz] finalize_episode failed: {e}")
            return {"ok": False, "stitched": False}


def load_agent(
    model_path=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False,
    expect_temporal_memory=True,
    expect_spatial_memory=True,
):
    device = f"cuda:{device}"
    assert model_path is not None

    model_folder = os.path.dirname(model_path)
    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path is not None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    if not use_input_place_with_mean:
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True
    exp_cfg.freeze()

    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path is not None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
    mvt_cfg.freeze()

    # Guard: eval-side memory ablation switches (run_server.sh TEMPORAL_MEMORY /
    # SPATIAL_MEMORY) must match the trained model's memory config.
    assert_eval_memory_matches_model(
        mvt_cfg, expect_temporal_memory, expect_spatial_memory,
        where="MemoryBench eval",
    )

    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()

    rvt = MVT(renderer_device=device, **mvt_cfg)
    # See note in train.py: ``image_resolution`` is dead in RVTAgent (stored,
    # never read). The live eval-time resolution is set by ``RLBenchEnv(
    # image_size=...)`` in client.py.
    agent = bridgevla_agent.RVTAgent(
        network=rvt.to(device),
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 1000)),
        # Train-time-only gate; eval never reads the collision slot on this
        # bench. Passed so ``agent.predict_collision`` honestly reflects the
        # checkpoint's own exp_cfg.yaml (old ckpts report True).
        predict_collision=bool(getattr(exp_cfg, "predict_collision", True)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )
    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()
    print("Agent ready.")
    return agent
