#Copy from https://github.com/NVlabs/RVT/blob/master/rvt/utils/rlbench_planning.py
import numpy as np
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    Scene,
)
from rlbench.action_modes.action_mode import MoveArmThenGripper

# Post-motion settle parameters. path.step() returning done only means the
# *commanded* Reflexxes trajectory finished; the PID-tracked joints can still
# be converging on the final targets. Before the gripper actuates (and before
# TaskEnvironment.step captures the observation for the next inference) we
# step the sim until every arm joint has physically stopped.
SETTLE_MAX_VEL = 0.01   # rad/s; all |joint velocity| below this == stopped
SETTLE_MAX_STEPS = 100  # safety cap (~5 s sim time at the 50 ms default dt)
# NOTE: there used to be an extra fixed hold (25 scene.step() == 1 s) after each
# keypose, meant to let free objects settle before the next observation. Removed
# after measuring it: on the full 18-task suite (model_130, 25 ep/task) mean SR
# was 90.4 (1 s) / 91.3 + 92.7 (two 0.2 s repeats) / 90.0 (no hold) — the spread
# between two *identical* configs exceeded the spread across configs. place_cups,
# the one task that looked hurt, turned out to score 52/64/76/92 across four
# no-hold repeats (16 of its 25 episodes flip run-to-run), i.e. pure noise from
# the sampling-based planner. Arm stillness is what actually matters and
# _wait_arm_stopped guarantees it.


class MoveArmThenGripper2(MoveArmThenGripper):
    """MoveArmThenGripper that understands the RVT/BridgeVLA 9-D action
    layout ``[pose(7), gripper(1), ignore_collisions(1)]``.

    The rjgpinel fork's generic MoveArmThenGripper treats everything after the
    arm pose as the gripper action, so the trailing predicted collision flag is
    handed to the Discrete gripper as a shape-(2,) action -> InvalidActionError
    on every step (episodes die after 1 step with score 0). Here the collision
    flag is routed to the arm planner's ``ignore_collisions`` instead (the arm
    action mode, EndEffectorPoseViaPlanning2, accepts it), and the gripper gets
    exactly its 1-D action.

    It also enforces settled execution: the gripper only actuates once the arm
    has physically stopped (see SETTLE_*)."""

    def _wait_arm_stopped(self, scene: Scene):
        for _ in range(SETTLE_MAX_STEPS):
            if np.max(np.abs(scene.robot.arm.get_joint_velocities())) \
                    < SETTLE_MAX_VEL:
                break
            scene.step()

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = int(np.prod(self.arm_action_mode.action_shape(scene)))
        grip_act_size = int(np.prod(self.gripper_action_mode.action_shape(scene)))
        arm_action = np.array(action[:arm_act_size])
        gripper_action = np.array(
            action[arm_act_size:arm_act_size + grip_act_size])
        tail = action[arm_act_size + grip_act_size:]
        # Predicted ignore_collisions flag (0/1); default to True (ignore) when
        # the action carries no collision element, matching the fork default.
        ignore_collisions = bool(int(round(float(tail[0])))) if len(tail) else True
        observations = self.arm_action_mode.action(
            scene, arm_action, ignore_collisions)
        self._wait_arm_stopped(scene)
        self.gripper_action_mode.action(scene, gripper_action)
        # The gripper attach/release can nudge the arm; re-confirm it is
        # stationary so the scored observation sees a settled scene.
        self._wait_arm_stopped(scene)
        return observations


class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )

        super().action(scene, action, ignore_collisions)
