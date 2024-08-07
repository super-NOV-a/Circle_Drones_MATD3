import itertools
import numpy as np
import pybullet as p
from gym_pybullet_drones.new_envs.CircleRL_Camera_Aviary import CircleRLCameraAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


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
                 test: bool = False,  # 若测试，则实例化对象时给定初始位置
                 discrete: bool = False
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
                         discrete=discrete
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
        if self.OBS_TYPE == ObservationType.RGB:
            states = {i: self._getDroneStateVector(i, self.need_target) for i in range(self.NUM_DRONES)}
            rgbs = {i: self._getDroneImages(i, False)[0] for i in range(self.NUM_DRONES)}
            dis_to_target = np.array(
                [np.linalg.norm(states[idx]['target_pos_dis'][:2]) for idx in range(self.NUM_DRONES)])

            rewards = self.compute_rgb_rewards(states, rgbs, dis_to_target, self.NUM_DRONES,
                                               self.previous_dis_to_target)
            self.previous_dis_to_target = dis_to_target  # 更新前一步的目标距离
            return rewards
        else:  # 观测为RGB
            print('[ERROR] in CircleSpread_Camera._computeReward(), Obs type is not RGB ')
            return -1

    def compute_rgb_rewards(self, states, rgbs, dis_to_target, num_drones, previous_dis_to_target):
        rewards = np.zeros(num_drones)
        for i in range(num_drones):
            get_target, rgb_reward = self.rgb_reward(rgbs[i])
            rewards[i] += rgb_reward
            if get_target:
                rewards[i] -= self.yaw_rate_reward(states[i]['ang_vel'][2])  # 观测到惩罚旋转
                rewards[i] += self.improvement_reward(dis_to_target[i], previous_dis_to_target[i])
            else:
                rewards[i] += self.yaw_rate_reward(states[i]['ang_vel'][2])  # 没观测到鼓励旋转
        return rewards

    def yaw_rate_reward(self, yaw_rate):
        if np.abs(yaw_rate) > .1:
            return 2
        else:
            return 0

    def rgb_reward(self, rgb):
        _, _, yellow_pixel, red_pixel = self.detect_yellow_red_regions(rgb)
        if yellow_pixel > 2:
            return True, yellow_pixel * 2 + red_pixel
        return False, 0

    def detect_yellow_red_regions(self, image):
        yellow_min = np.array([180, 180, 0], dtype=np.uint8)
        yellow_max = np.array([255, 255, 70], dtype=np.uint8)
        red_min = np.array([180, 0, 0], dtype=np.uint8)
        red_max = np.array([255, 70, 70], dtype=np.uint8)
        img_rgb = image[:, :, :3]
        yellow_mask = np.all((img_rgb >= yellow_min) & (img_rgb <= yellow_max), axis=-1)
        red_mask = np.all((img_rgb >= red_min) & (img_rgb <= red_max), axis=-1)
        yellow_pixel_count = np.sum(yellow_mask)
        red_pixel_count = np.sum(red_mask)
        return yellow_mask, red_mask, yellow_pixel_count, red_pixel_count

    def detect_yellow_region(self, image):
        yellow_min = np.array([160, 160, 0], dtype=np.uint8)
        yellow_max = np.array([255, 255, 100], dtype=np.uint8)
        img_rgb = image[:, :, :3]  # 提取RGB通道
        yellow_mask = np.all((img_rgb >= yellow_min) & (img_rgb <= yellow_max), axis=-1)  # 识别黄色区域
        yellow_pixel_count = np.sum(yellow_mask)  # 计算黄色区域的像素数
        return yellow_pixel_count

    def improvement_reward(self, dis_to_target, previous_dis_to_target):
        if dis_to_target < previous_dis_to_target:
            return 10  # 你可以调整奖励的数值
        return 0

    ################################################################################
    def _computeTerminated(self):
        """Computes the current done, punish value.

        Returns
        -------
        list, list
            A list indicating whether each drone is done and a list indicating whether each drone is punished.
        """
        dones = [False for _ in range(self.NUM_DRONES)]
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            dones = [True for _ in range(self.NUM_DRONES)]
        return dones

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
