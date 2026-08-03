# Copy from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/utils/custom_rlbench_env.py
from bridgevla.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv
from pyrep.objects.shape import Shape


# --- Missing-asset (non-renderable) fix -------------------------------------
# The eval RLBench fork's .ttm files ship several static/target visual meshes
# with the "renderable" special-property cleared. RLBench builds RGB *and*
# depth/point-cloud from vision sensors, which only capture renderable objects,
# so these meshes emit no depth -> they vanish from the point cloud the model
# consumes (and from the cinematic video), while the PerAct *training data* was
# generated with them renderable. That train/eval mismatch makes the affected
# tasks fail (e.g. light_bulb_in shows no lamp base; put_item_in_drawer /
# put_money_in_safe drop the grasp target entirely).
#
# Verified via segmentation-mask presence across all 18 PerAct eval tasks: the
# 5 tasks below are the only ones affected, and every listed object IS present
# in the training data -> forcing it renderable RESTORES parity. Keyed by
# Task.get_name() (snake_case). `is_renderable()` alone over-reports (physics
# parents with renderable visual children), so this list is mask-verified, not
# flag-derived.
GHOST_RENDERABLE_FIX = {
    "put_item_in_drawer": ["item"],
    "put_money_in_safe": ["dollar_front_visual", "dollar_back_visual"],
    "light_bulb_in": ["lamp_base", "lamp_screw"],
    "place_cups": ["place_cups_holder_spoke3"],
    "place_shape_in_shape_sorter": ["shape_sorter_top"],
}


class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)

    def _force_renderable_ghosts(self):
        """Re-enable the renderable flag on the current task's ghost meshes.

        Must run after each (re)load of the task model, since the flag lives in
        the freshly-imported .ttm. Returns the number of objects toggled so the
        caller can re-capture the initial observation only when needed (the
        observation returned by reset/reset_to_demo was rendered *before* this
        fix, so it would otherwise miss the ghost at step 0)."""
        try:
            task_name = self._task._task.get_name()
        except Exception:
            return 0
        n = 0
        for obj_name in GHOST_RENDERABLE_FIX.get(task_name, []):
            try:
                Shape(obj_name).set_renderable(True)
                n += 1
            except Exception:
                # Object may be absent for some variation; skip silently.
                pass
        return n

    def reset(self) -> dict:
        super().reset()
        if self._force_renderable_ghosts() > 0:
            # Re-render the step-0 observation now that the ghost is visible.
            self._previous_obs_dict = self.extract_obs(
                self._task.get_observation())
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        # Restore renderability of ghost meshes on the freshly-loaded model,
        # then re-capture so the step-0 observation actually contains them.
        if self._force_renderable_ghosts() > 0:
            obs = self._task.get_observation()

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
