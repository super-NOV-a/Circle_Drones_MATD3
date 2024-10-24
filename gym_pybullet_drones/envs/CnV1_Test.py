import numpy as np
from gym_pybullet_drones.envs.CnV1RL_Test import CnV1RL_Test
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class CnV1_Test(CnV1RL_Test):
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
                 follow_distance: float = 1.0,  # 跟踪敌机的距离 胜利条件
                 acctack_distance: float = .1,  # 打击敌机的距离 胜利条件
                 keep_distance: float = .1,     # 不碰撞距离 成功条件
                 all_axis: float = 2.,     # XY范围
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
                         all_axis=all_axis,
                         )

        self.EPISODE_LEN_SEC = 100
        self.previous_dis_to_target = np.zeros(num_drones)  # 初始化前一步的目标距离
        self.collision = False
        self.follow_distance = follow_distance
        self.acctack_distance = acctack_distance
        self.keep_distance = keep_distance

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

        # 队友保持距离与碰撞惩罚
        if self.NUM_DRONES > 1:
            other_pos_dis = np.array([state['other_pos_dis'] for state in states.values()])
            dist_between_drones = other_pos_dis[:, 3::4]  # 获取距离
            rewards -= np.sum(100 * np.power(5, (-4 * dist_between_drones - 1)) - 0.2, axis=1)
        return rewards

    ################################################################################
    def _computeTerminated(self):
        dones = np.full(self.NUM_DRONES, False)
        states = {i: self._getDroneStateVector(i, with_target=True) for i in range(self.NUM_DRONES)}

        if self.NUM_DRONES > 1:
            other_pos_dis = np.array([state['other_pos_dis'] for state in states.values()])
            dist_between_drones = other_pos_dis[:, 3::4]  # 获取每个无人机之间的距离

            # 如果任意无人机之间距离小于 安全阈值，则失败（不成功），但不一定不胜利，两者条件不同
            if np.any(dist_between_drones <= self.keep_distance):
                self.collision = True

            dis_to_target = np.array([state['target_pos_dis'][3] for state in states.values()])
            # if np.all(dis_to_target <= self.follow_distance):  # 全部无人机都靠近目标
            dones = dis_to_target <= self.acctack_distance  # todo 修改done条件为靠近就有

        # done为打击胜利条件，fail为考虑碰撞的成功条件
        return dones.tolist(), self.collision

    ################################################################################

    def _computeTruncated(self):
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:  # step_counter/240>8 -> step_counter每step+8 单episode最大长度应该只有240
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years
