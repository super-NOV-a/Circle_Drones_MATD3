import itertools

import numpy as np

import pybullet as p
from gym_pybullet_drones.envs.LyyRLAviary import LyyRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class Spread3dAviary(LyyRLAviary):
    """Multi-agent RL problem: simple_spread in 3d."""

    ################################################################################

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

    ################################################################################
    import numpy as np

    def _computeReward(self):
        """Computes the current reward value.

        state = Dict
            (3,   4,    3,   3,    3,       3*n,             3*(n-1),         4)
            (pos, quat, rpy, vel, ang_vel, target_pos, other_pos, last_clipped_action)
        Returns
        -------
        list of float
            The reward for each drone.

        """
        states = {i: self._getDroneStateVector(i, with_target=True) for i in range(self.NUM_DRONES)}
        ret = [0 for _ in range(self.NUM_DRONES)]

        # 计算每个无人机到每个目标的距离
        distances_to_targets = np.array(
            [[np.linalg.norm(states[i]['target_pos'][j]) for j in range(self.NUM_DRONES)] for i in
             range(self.NUM_DRONES)])

        # 为每个无人机计算奖励
        for i in range(self.NUM_DRONES):
            # 最近的目标距离
            min_dist_to_target = min(distances_to_targets[i])
            ret[i] -= min_dist_to_target  # 每个无人机与最近目标的距离作为奖励，距离越小奖励越高

            # 具体最近目标的距离奖励
            target_index = np.argmin(distances_to_targets[i])
            target_pos = states[i]['target_pos'][0+target_index*3: 3+target_index*3]   # 最近的目标位置向量
            if np.abs(target_pos[2]) < 0.03:  # 高度差
                ret[i] += 1
                if np.abs(target_pos[2]) < 0.02:
                    ret[i] += 1
                    if np.abs(target_pos[2]) < 0.01:
                        ret[i] += 2
                if min_dist_to_target < 0.05:
                    ret[i] += 1
                    if min_dist_to_target < 0.02:
                        ret[i] += 2
                        if min_dist_to_target < 0.01:
                            ret[i] += 5
            # 碰撞惩罚
            if self.NUM_DRONES != 1:
                for j in range(self.NUM_DRONES-1):
                    dist_between_drones = np.linalg.norm(states[i]['other_pos'][0+j*3: 3+j*3])
                    if dist_between_drones < 0.13:  # 碰撞体积为圆柱形，0.12米为碰撞直径
                        ret[i] -= 10
        return ret

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
        out_of_bounds_count = 0

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i, True)
            x, y, z = state['pos']
            roll, pitch = state['rpy'][0], state['rpy'][1]

            # 检查出界
            if abs(x) > 1.5 or abs(y) > 1.5 or z > 1.5 or z < 0.05:
                punish[i] = 1.0
                out_of_bounds_count += 1

            if abs(x) > 2 or abs(y) > 2 or z > 1.5 or z < 0.05:
                # dones[i] = True  # 出太多就 done <- done而不结束 是否影响计算reward
                punish[i] = 2.0
                # self._resetDronePosition(i, [0, 0, 0.2])    # 出去后恢复？看看效果

            # 姿态惩罚
            if abs(roll) > 0.4 or abs(pitch) > 0.4:
                punish[i] = max(punish[i], 0.2)   # 未出界但是姿态不稳定

        # 出界过多,执行够次数 结束回合
        if out_of_bounds_count >= (self.NUM_DRONES + 1) / 2:
            dones = [True for _ in range(self.NUM_DRONES)]
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
