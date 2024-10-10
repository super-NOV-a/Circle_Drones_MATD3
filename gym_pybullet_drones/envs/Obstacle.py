import itertools
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.ObstacleRLAviary import ObstacleRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class Obstacle(ObstacleRLAviary):
    """Multi-agent RL problem: simple_spread in 3d."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 num_obstacle: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,  # 原本是 30
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 need_target: bool = False,
                 obs_with_act: bool = False,
                 # seed=0,
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
                         num_obstacle=num_obstacle,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         need_target=need_target,
                         obs_with_act=obs_with_act,
                         )

        self.EPISODE_LEN_SEC = 100
        self.previous_dis_to_target = np.zeros(num_drones)  # 初始化前一步的目标距离

    ################################################################################
    # def _computeReward(self):
    #     """计算当前的奖励值。
    #
    #     state = Dict
    #         (3,   4,    3,   3,    3,           4,            (n-1)*4,         4)
    #         (pos, quat, rpy, vel, ang_vel, target_pos_dis, other_pos_dis, last_clipped_action)
    #     Returns
    #     -------
    #     list of float
    #         每个无人机的奖励值。
    #     """
    #     states = {i: self._getDroneStateVector(i, with_target=True) for i in range(self.NUM_DRONES)}
    #     rewards = [0 for _ in range(self.NUM_DRONES)]
    #
    #     # 计算目标点距离
    #     dis_to_target = np.array([np.linalg.norm(states[idx]['target_pos_dis'][:2]) for idx in range(self.NUM_DRONES)])
    #     h_to_target = np.array([states[idx]['target_pos_dis'][2] for idx in range(self.NUM_DRONES)])
    #     velocity = np.array([np.linalg.norm(states[idx]['vel'][-1]) for idx in range(self.NUM_DRONES)])
    #
    #     # 为每个无人机计算奖励
    #     for i in range(self.NUM_DRONES):
    #         # 鼓励在目标附近的小范围内移动
    #         if dis_to_target[i] < 0.1:
    #             rewards[i] += 15  # 当距离目标很近时，给予较大的奖励
    #         elif dis_to_target[i] < 0.2:
    #             rewards[i] += 8
    #         elif dis_to_target[i] < 0.5:
    #             rewards[i] += 4  # 当距离目标较近时，给予中等奖励
    #         elif dis_to_target[i] < 1.0:
    #             rewards[i] += 1
    #         else:
    #             rewards[i] -= 0.1  # 距离目标较远时，给予惩罚
    #
    #         # 适度减少速度惩罚
    #         rewards[i] -= 0.1 * velocity[i]
    #         if dis_to_target[i] < 0.3:  # 靠近目标需要更小的速度
    #             rewards[i] -= 1 * velocity[i]
    #
    #         # 鼓励保持在目标的合适高度范围内
    #         if np.abs(h_to_target[i]) < 0.05:
    #             rewards[i] += 5  # 高度接近目标时，给予较大的奖励
    #         elif np.abs(h_to_target[i]) < 0.1:
    #             rewards[i] += 3  # 高度较接近目标时，给予中等奖励
    #         elif np.abs(h_to_target[i]) < 0.4:
    #             rewards[i] += 1  # 高度接近目标时，给予微弱奖励
    #         else:
    #             rewards[i] -= 0.1  # 高度较远时，给予惩罚
    #
    #         # 如果这一step的dis_to_target比上一step小，则给一个正的奖励
    #         if dis_to_target[i] < self.previous_dis_to_target[i]:
    #             rewards[i] += 1  # 你可以调整奖励的数值
    #
    #     # 队友保持距离与碰撞惩罚
    #     if self.NUM_DRONES != 1:
    #         for i in range(self.NUM_DRONES):
    #             rewards_i = 0  # 临时存储奖励值，减少对 rewards 列表的访问次数
    #             state_i = states[i]['other_pos_dis']
    #             for j in range(self.NUM_DRONES - 1):
    #                 dist_between_drones = state_i[j * 4 + 3]  # 获取距离
    #                 # delta_h_drones = state_i[j * 4 + 2]  # 获取高度差
    #                 if dist_between_drones < 0.13:
    #                     rewards_i -= 1
    #                 if dist_between_drones > 1:
    #                     rewards_i -= 1
    #             rewards[i] += rewards_i  # 将临时存储的奖励值加到 rewards 中
    #
    #     self.previous_dis_to_target = dis_to_target  # 更新前一步的目标距离
    #     return rewards

    def _computeReward(self):
        """
        计算当前的奖励值。

        state = Dict
        (3,   4,    3,   3,    3,           4,            (n-1)*4,         4)
        (pos, quat, rpy, vel, ang_vel, target_pos_dis, other_pos_dis, last_clipped_action)
        Returns
        -------
        list of float
        每个无人机的奖励值。
        """
        states = {i: self._getDroneStateVector(i, with_target=True) for i in range(self.NUM_DRONES)}
        rewards = np.zeros(self.NUM_DRONES)  # 使用 NumPy 数组
        dis_to_target = np.array([state['target_pos_dis'] for state in states.values()])  # 4
        velocity = np.array([state['vel'] for state in states.values()])  # 3
        re_pos = np.array([state['pos'] for state in states.values()])  # 障碍距离倒数和
        v = np.linalg.norm(velocity, axis=1)  # 计算速度的 L2 范数

        # 计算奖励
        rewards += 15 * np.power(20, -dis_to_target[:, -1])  # 距离奖励
        rewards += np.sum(velocity * dis_to_target[:, :3], axis=1) / (v * dis_to_target[:, -1])
        rewards += 3 * np.power(20, -np.abs(dis_to_target[:, 2]))  # 高度奖励
        # rewards -= 0.1* np.linalg.norm(velocity - self.last_v, axis=1) / np.where(v > 0, v, 1)  # 加速度惩罚
        angular_velocity = np.linalg.norm(np.array([state['ang_vel'] for state in states.values()]), axis=1)
        rewards -= 0.5 * angular_velocity  # 角速度惩罚
        rewards -= 20*np.power(re_pos[:, 2], 2)-0.2  # 距离障碍距离

        # 队友保持距离与碰撞惩罚
        if self.NUM_DRONES > 1:
            other_pos_dis = np.array([state['other_pos_dis'] for state in states.values()])
            dist_between_drones = other_pos_dis[:, 3::4]  # 获取距离
            rewards -= np.sum(50 * np.power(5, (-4 * dist_between_drones - 1)) - 0.2, axis=1)
        # self.last_v = velocity.copy()  # 直接更新 last_v

        np.linalg.norm(re_pos)
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

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i, True)
            x, y, z = self.pos[i]
            dis = state['target_pos_dis'][3]
            roll, pitch, _ = state['rpy']

            # 检查出界
            if z > 4 or z < 0 or dis > 10:
                punish[i] = 10

            # 姿态惩罚
            if abs(roll) > 0.4 or abs(pitch) > 0.4:
                punish[i] = max(punish[i], 1)   # 未出界但是姿态不稳定

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
