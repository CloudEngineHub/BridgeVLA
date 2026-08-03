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
Adapted from https://github.com/vlc-robot/robot-3dlotus/blob/main/challenges/actioner.py

'''
import os

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.utils.rvt_utils import load_agent as load_agent_state
from bridgevla.utils.memory_switches import assert_eval_memory_matches_model
from bridgevla.mvt.mvt import MVT
from utils.peract_utils_gembench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)

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
        print("your model path:")
        print(model_path)
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
        '''Args:
            taskvar: str, 'task+variation'
            episode_id: int
            step_id: int, [0, 25]
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv
            visualize: bool, if True save rendered views / heatmaps / point cloud
            visualize_episode_dir: per-episode save root; per-step subdir is appended inside agent.act
        '''
        # Episode boundary: clear the agent's MemoryBank so anchor / history
        # tokens from the previous episode don't leak into this one. Required
        # whenever ``memory.enabled=True``; harmless no-op otherwise. The
        # GemBench client/server protocol has no /reset hook, so we detect the
        # boundary from ``step_id == 0`` here (matches YARR's
        # rollout_generator, which calls ``agent.reset()`` at episode start).
        if step_id == 0:
            self.agent.reset()

        for idx, cam in enumerate(self.agent.cameras):
            obs_state_dict[f"{cam}_rgb"] = np.transpose(obs_state_dict["rgb"][idx], [2, 0, 1])[None]
            obs_state_dict[f"{cam}_point_cloud"] = np.transpose(obs_state_dict["pc"][idx], [2, 0, 1])[None]
        
        del obs_state_dict["rgb"]
        del obs_state_dict["pc"]
        del obs_state_dict['arm_links_info']
        del obs_state_dict['depth']
        del obs_state_dict['gripper']
        
        for k, v in obs_state_dict.items():
            if isinstance(v, np.ndarray):
                obs_state_dict[k] = torch.from_numpy(v).to(self.agent._device)
            elif isinstance(v, list):
                obs_state_dict[k] = torch.tensor(v).to(self.agent._device)
            elif isinstance(v, torch.Tensor):
                obs_state_dict[k] = v.to(self.agent._device)    
            obs_state_dict[k] = obs_state_dict[k].unsqueeze(0)
        obs_state_dict["language_goal"] =   [[[instruction]]]
        # torch.no_grad: RVTAgent.act is NOT decorated with @torch.no_grad, so
        # without this the two-stage 3B forward builds an autograd graph and can
        # OOM at eval. Matches finetune/memoryBench/actioner.py.
        #
        # return_gembench_action=True -> 8-D [wpt(3), quat(4), grip(1)]. The
        # collision slot is dropped unconditionally: GemBench's env is built
        # with EndEffectorPoseViaPlanning(collision_checking=False) and the
        # stock MoveArmThenGripper.action() takes no per-call ignore_collisions,
        # so the planner runs with ignore_collisions=True on every step by
        # construction. This is the fixed convention for this bench — nothing
        # here is conditional on the checkpoint or on predict_collision.
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
        '''Episode-end hook (called via the server /finalize route). Stitch the
        just-finished episode's per-step overlay tri-views into per-stage grids
        (grid_mvt1.png / grid_mvt2.png). No-op when overlay viz wasn't active.'''
        try:
            return self.agent.finalize_eval_viz()
        except Exception as e:
            print(f"[GemBench eval viz] finalize_episode failed: {e}")
            return {"ok": False, "stitched": False}
    

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
    if exp_cfg_path != None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

    # NOTE: to not use place_with_mean in evaluation
    # needed for rvt-1 but not rvt-2
    if not use_input_place_with_mean:
        # for backward compatibility
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True

    exp_cfg.freeze()


    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path != None:
        mvt_cfg.merge_from_file(mvt_cfg_path)
    else:
        mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

    mvt_cfg.freeze()

    # Guard: eval-side memory ablation switches (run_server.sh TEMPORAL_MEMORY /
    # SPATIAL_MEMORY) must match the trained model's memory config.
    assert_eval_memory_matches_model(
        mvt_cfg, expect_temporal_memory, expect_spatial_memory,
        where="GemBench server",
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
        # Train-time-only gate; eval never reads the collision slot on this
        # bench (see the act() call below). Passed so ``agent.predict_collision``
        # honestly reflects the checkpoint's own exp_cfg.yaml.
        predict_collision=bool(getattr(exp_cfg, "predict_collision", True)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )


    agent.build(training=False, device=device)
    # use_view_logvar branch is deferred (see bridgevla.mvt.view_logvar);
    # the agent constructor would have raised NotImplementedError already
    # if the flag were True.
    load_agent_state(model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent


