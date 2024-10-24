import numpy as np
from gym_pybullet_drones.envs.C3V1_CommRLAviary import C3V1_CommRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class C3V1_Comm(C3V1_CommRLAviary):
    """Multi-agent RL problem: 3 VS 1 3d."""
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
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
                 comm_level: int = 1,
                 ):
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
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
                         comm_level=comm_level,
                         )

        self.EPISODE_LEN_SEC = 100
        self.previous_dis_to_target = np.zeros(num_drones)  # 初始化前一步的目标距离

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
        rewards = np.zeros(self.NUM_DRONES)
        dis_to_target = np.array([state['target_pos_dis'] for state in states.values()])  # 4
        velocity = np.array([state['vel'] for state in states.values()])  # 3
        v = np.linalg.norm(velocity, axis=1)  # 计算速度的 L2 范数

        rewards += 10 * np.power(20, -dis_to_target[:, -1])  # 距离目标奖励
        rewards -= 0.1 * v  # 速度惩罚
        rewards += np.sum(velocity * dis_to_target[:, :3], axis=1) / (v * dis_to_target[:, -1])  # 相似度奖励
        rewards += 3 * np.power(20, -np.abs(dis_to_target[:, 2]))  # 高度奖励
        # rewards -= 0.1* np.linalg.norm(velocity - self.last_v, axis=1) / np.where(v > 0, v, 1)  # 加速度惩罚
        # angular_velocity = np.linalg.norm(np.array([state['ang_vel'] for state in states.values()]), axis=1)
        # rewards -= 0.5 * angular_velocity  # 角速度惩罚

        # 队友保持距离与碰撞惩罚
        if self.NUM_DRONES > 1:
            other_pos_dis = np.array([state['other_pos_dis'] for state in states.values()])
            dist_between_drones = other_pos_dis[:, 3::4]  # 获取距离
            rewards -= np.sum(100 * np.power(5, (-4 * dist_between_drones - 1)) - 0.2, axis=1)
        return rewards

    ################################################################################
    def _computeTerminated(self):
        dones = [False for _ in range(self.NUM_DRONES)]
        punish = [0.0 for _ in range(self.NUM_DRONES)]  # Use a floating-point value for dynamic punish
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i, True)
            x, y, z = state['pos']
            dis = state['target_pos_dis'][3]
            roll, pitch, _ = state['rpy']

            if dis < 0.05:
                dones[i] = True
                punish[i] -= 20
            if z > 4 or z < 0 or dis > 10:  # 检查出界
                punish[i] = 10
            if abs(roll) > 0.4 or abs(pitch) > 0.4:     # 姿态惩罚
                punish[i] = max(punish[i], 1)   # 未出界但是姿态不稳定
        return dones, punish

    ################################################################################

    def _computeTruncated(self):
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:  # step_counter/240>8 -> step_counter每step+8 单episode最大长度应该只有240
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years
