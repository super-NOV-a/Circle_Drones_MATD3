import itertools
import numpy as np
import pybullet as p
from gym_pybullet_drones.new_envs.CircleRL_Camera_Aviary import CircleRLCameraAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def distance_to_circle_reward(dis_to_circle):
    if dis_to_circle < 0.1:
        return 15  # 当距离目标很近时，给予较大的奖励
    elif dis_to_circle < 0.2:
        return 10
    elif dis_to_circle < 0.5:
        return 5  # 当距离目标较近时，给予中等奖励
    elif dis_to_circle < 1.0:
        return 2
    else:
        return -0.5  # 距离目标较远时，给予惩罚


def velocity_penalty(velocity, dis_to_circle):
    penalty = -0.1 * velocity
    if dis_to_circle < 0.3:  # 靠近目标需要更小的速度
        penalty -= 1 * velocity
    return penalty


def distance_to_target_penalty(dis_to_target):
    if dis_to_target < 0.13:
        return -5
    return 0


def height_reward(h_to_target):
    if np.abs(h_to_target) < 0.05:
        return 10  # 高度接近目标时，给予较大的奖励
    elif np.abs(h_to_target) < 0.1:
        return 5  # 高度较接近目标时，给予中等奖励
    elif np.abs(h_to_target) < 0.2:
        return 1  # 高度接近目标时，给予微弱奖励
    else:
        return -0.5  # 高度较远时，给予惩罚


def improvement_reward(dis_to_target, previous_dis_to_target):
    if dis_to_target < previous_dis_to_target:
        return 1  # 你可以调整奖励的数值
    return 0


def collision_penalty(states, i, num_drones):
    rewards_i = 0  # 临时存储奖励值，减少对 rewards 列表的访问次数
    state_i = states[i]['other_pos_dis']
    for j in range(num_drones - 1):
        dist_between_drones = state_i[j * 4 + 3]  # 获取距离
        if dist_between_drones < 0.13:
            rewards_i -= 1
    return rewards_i


def rgb_reward(rgb):
    pass
    return 0


def compute_rewards(states, dis_to_circle, velocity, dis_to_target, h_to_target, num_drones, previous_dis_to_target):
    rewards = np.zeros(num_drones)
    for i in range(num_drones):
        rewards[i] += distance_to_circle_reward(dis_to_circle[i])
        rewards[i] += velocity_penalty(velocity[i], dis_to_circle[i])
        rewards[i] += distance_to_target_penalty(dis_to_target[i])
        rewards[i] += height_reward(h_to_target[i])
        rewards[i] += improvement_reward(dis_to_target[i], previous_dis_to_target[i])
        # 队友保持距离与碰撞惩罚
        if num_drones != 1:
            rewards[i] += collision_penalty(states, i, num_drones)
    return rewards


def compute_rgb_rewards(states, rgbs, dis_to_circle, velocity, dis_to_target, h_to_target, num_drones, previous_dis_to_target):
    rewards = np.zeros(num_drones)
    for i in range(num_drones):
        rewards[i] += rgb_reward(rgbs[i])
        rewards[i] += distance_to_circle_reward(dis_to_circle[i])
        rewards[i] += distance_to_target_penalty(dis_to_target[i])
        rewards[i] += height_reward(h_to_target[i])
        rewards[i] += improvement_reward(dis_to_target[i], previous_dis_to_target[i])
        # 队友保持距离与碰撞惩罚
        if num_drones != 1:
            rewards[i] += collision_penalty(states, i, num_drones)
    return rewards


class CircleCameraAviary(CircleRLCameraAviary):
    """Multi-agent RL problem: simple_spread in 3d."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 need_target: bool = False,
                 obs_with_act: bool = False,
                 test=False  # 若测试，则实例化对象时给定初始位置
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         need_target=need_target,
                         obs_with_act=obs_with_act,
                         test=test,
                         )

        self.EPISODE_LEN_SEC = 100
        self.previous_dis_to_target = np.zeros(num_drones)  # 初始化前一步的目标距离

    ################################################################################
    import numpy as np

    def _computeReward(self):
        """计算当前的奖励值。

        state = Dict
            (3,   4,    3,   3,    3,           4,            (n-1)*4,         4)
            (pos, quat, rpy, vel, ang_vel, target_pos_dis, other_pos_dis, last_clipped_action)
        Returns
        -------
        list of float
            每个无人机的奖励值。

        """
        if self.OBS_TYPE != ObservationType.RGB:
            states = {i: self._getDroneStateVector(i, self.need_target) for i in range(self.NUM_DRONES)}
            # 计算目标点距离
            dis_to_target = np.array([np.linalg.norm(states[idx]['target_pos_dis'][:2]) for idx in range(self.NUM_DRONES)])
            h_to_target = np.array([states[idx]['target_pos_dis'][2] for idx in range(self.NUM_DRONES)])
            dis_to_circle = np.abs(dis_to_target - 0.4)  # 0.4 是期望保持的距离
            velocity = np.array([np.linalg.norm(states[idx]['vel'][-1]) for idx in range(self.NUM_DRONES)])

            rewards = compute_rewards(states, dis_to_circle, velocity, dis_to_target, h_to_target, self.NUM_DRONES, self.previous_dis_to_target)
            self.previous_dis_to_target = dis_to_target  # 更新前一步的目标距离
            return rewards
        else:  # 观测为RGB
            states = {i: self._getDroneStateVector(i, self.need_target) for i in range(self.NUM_DRONES)}
            rgbs = {i: self._getDroneImages(i, False)[0] for i in range(self.NUM_DRONES)}
            dis_to_target = np.array([np.linalg.norm(states[idx]['target_pos_dis'][:2]) for idx in range(self.NUM_DRONES)])
            h_to_target = np.array([states[idx]['target_pos_dis'][2] for idx in range(self.NUM_DRONES)])
            dis_to_circle = np.abs(dis_to_target - 0.4)  # 0.4 是期望保持的距离
            velocity = np.array([np.linalg.norm(states[idx]['vel'][-1]) for idx in range(self.NUM_DRONES)])

            rewards = compute_rgb_rewards(states, rgbs, dis_to_circle, velocity, dis_to_target, h_to_target, self.NUM_DRONES, self.previous_dis_to_target)
            self.previous_dis_to_target = dis_to_target  # 更新前一步的目标距离
            return rewards

    ################################################################################
    def _computeTerminated(self):
        """Computes the current done, punish value.

        Returns
        -------
        list, list
            A list indicating whether each drone is done and a list indicating whether each drone is punished.
        """
        dones = [False for _ in range(self.NUM_DRONES)]
        punish = [0.0 for _ in range(self.NUM_DRONES)]  # Use a floating-point value for dynamic punish
        if self.OBS_TYPE != ObservationType.RGB:
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i, self.need_target)
                x, y, z = state['pos']
                dis = state['target_pos_dis'][3]
                roll, pitch, _ = state['rpy']
                # 检查出界
                if z > 3 or z < 0 or dis > 10:
                    punish[i] = 10
                # 姿态惩罚
                if abs(roll) > 0.4 or abs(pitch) > 0.4:
                    punish[i] = max(punish[i], 3)  # 未出界但是姿态不稳定
            if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
                dones = [True for _ in range(self.NUM_DRONES)]
            return dones, punish

        else:
            if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
                dones = [True for _ in range(self.NUM_DRONES)]
            return dones, punish

    def _resetDronePosition(self, drone_idx, new_position):
        """Resets the position of the specified drone.
            NOT USED  !!!!
        Parameters
        ----------
        drone_idx : int
            The index of the drone to reset.
        new_position : list
            The new position to reset the drone to.
        """
        new_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.DRONE_IDS[drone_idx], new_position, new_orientation,
                                          physicsClientId=self.CLIENT)

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:  # step_counter/240>8 -> step_counter每step+8 单episode最大长度应该只有240
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years
