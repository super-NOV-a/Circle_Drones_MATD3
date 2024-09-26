import numpy as np
from gym_pybullet_drones.envs.C3V1RLAviary import C3V1RLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class C3V1(C3V1RLAviary):
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
        rewards = [0 for _ in range(self.NUM_DRONES)]

        # 计算目标点距离
        dis_to_target = np.array([states[idx]['target_pos_dis'] for idx in range(self.NUM_DRONES)])     # 4
        velocity = np.array([states[idx]['vel'] for idx in range(self.NUM_DRONES)])     # 3
        v = np.array([np.linalg.norm(states[idx]['vel']) for idx in range(self.NUM_DRONES)])

        # 为每个无人机计算奖励
        for i in range(self.NUM_DRONES):
            # 鼓励在目标附近的小范围内移动
            rewards[i] += 10*pow(20, -dis_to_target[i][-1])  # 使用指数函数计算距离奖励 10*20^{-x}
            # 适度减少速度惩罚
            rewards[i] -= 0.1 * v[i]
            # 根据相似度调整奖励
            cos_similarity = np.dot(velocity[i][:3], dis_to_target[i][:3]) / (v[i] * dis_to_target[i][-1])
            rewards[i] += cos_similarity * 5  # 相似度越高，奖励越大
            # 鼓励保持在目标的合适高度范围内
            rewards[i] += 10*pow(20, -np.abs(dis_to_target[i][2]))  # 使用指数函数计算高度奖励 3*20^{-x}
            # 队友保持距离与碰撞惩罚
            if self.NUM_DRONES != 1:
                state_i = states[i]['other_pos_dis']
                for j in range(self.NUM_DRONES - 1):
                    dist_between_drones = state_i[j * 4 + 3]  # 获取距离
                    if dist_between_drones < 0.15:
                        rewards[i] -= 100*pow(5, (-4*dist_between_drones-1))  # 50*5^{-4x-1}
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

            # 检查出界
            if z > 4 or z < 0 or dis > 10:
                punish[i] = 10

            # 姿态惩罚
            if abs(roll) > 0.4 or abs(pitch) > 0.4:
                punish[i] = max(punish[i], 1)   # 未出界但是姿态不稳定

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:    # step_counter 最大是8 * 1000（看设置的）
            dones = [True for _ in range(self.NUM_DRONES)]

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
