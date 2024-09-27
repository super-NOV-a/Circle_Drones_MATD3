import os
import numpy as np
import pybullet as p
import torch
from gymnasium import spaces
from collections import deque

from torch.distributions import Categorical

from gym_pybullet_drones.new_envs.CircleBase_Camera_Aviary import CircleBaseCameraAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class CircleRLCameraAviary(CircleBaseCameraAviary):
    """Lyy Base single and multi-agent environment class for reinforcement learning.
        Note : 无人机最优的位置应该是一个环上!!!
    """

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
                 test: bool = False, # 若测试，则实例化对象时给定初始位置
                 discrete: bool = False
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        self.discrete = discrete  # 动作是否为离散
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(2)  # 动作缓存在此指定，原本是根据控制频率给出的
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.MIXED, ActionType.V_YAW]:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in LyyRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         gui=gui,
                         record=record,
                         obstacles=True,  # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False,  # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         need_target=need_target,
                         obs_with_act=obs_with_act,
                         test=test,
                         discrete=discrete,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL or act == ActionType.MIXED or act == ActionType.V_YAW:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides LyyBaseAviary's method.

        """
        if self.Test:
            p.loadURDF("block.urdf",
                       [1, 0, .2],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("block.urdf",
                       [1, 0, .3],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            if self.OBS_TYPE == ObservationType.RGB:
                p.loadURDF("block.urdf",
                           [1, 0, .1],
                           p.getQuaternionFromEuler([0, 0, 0]),
                           physicsClientId=self.CLIENT
                           )
                p.loadURDF("cube_small.urdf",
                           [0, 1, .1],
                           p.getQuaternionFromEuler([0, 0, 0]),
                           physicsClientId=self.CLIENT
                           )
                p.loadURDF("duck_vhacd.urdf",
                           [-1, 0, .1],
                           p.getQuaternionFromEuler([0, 0, 0]),
                           physicsClientId=self.CLIENT
                           )
                p.loadURDF("teddy_vhacd.urdf",
                           [0, -1, .1],
                           p.getQuaternionFromEuler([0, 0, 0]),
                           physicsClientId=self.CLIENT
                           )
            else:
                pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        list
            A list of action spaces for each drone, either Box or Discrete.

        """
        if self.discrete:
            # Define discrete action space with 8 possible actions
            action_space = [8 for _ in range(self.NUM_DRONES)]  # 此处表示为离散值
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, 8)))
        else:
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                size = 4
            elif self.ACT_TYPE == ActionType.PID:
                size = 3
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                size = 1
            elif self.ACT_TYPE == ActionType.V_YAW:
                size = 5
            else:
                print("[ERROR] in LyyRLAviary._actionSpace()")
                exit()

            act_lower_bound = np.array(-1 * np.ones(size))
            act_upper_bound = np.array(+1 * np.ones(size))

            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))

            action_space = [spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32) for _ in
                            range(self.NUM_DRONES)]

        return action_space

    ################################################################################
    def _preprocessAction(self, action, detected_num):
        """Pre-processes the action passed to `.step()` into motors' 4 RPMs. Returns (NUM_DRONES, 4) """
        safe_penalty = np.zeros(self.NUM_DRONES)  # 如果有不安全的动作则保证动作安全且减小奖励
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))  # 最终计算结果为rpm
        action_tensor = torch.tensor(action, dtype=torch.float32)  # (num_drones, 8)

        for k in range(self.NUM_DRONES):  # 对每架无人机进行处理
            if detected_num[k] >= 2:  # 黄色像素大于2，移动动作为主
                action_tensor[k, 4:8] = 0  # 仅保留移动动作的概率，将旋转动作的概率设为0

            action_probs = action_tensor[k] / action_tensor[k].sum(dim=-1, keepdim=True)  # 重新归一化动作概率
            action_id = Categorical(probs=action_probs).sample().item()  # 根据调整后的概率分布采样动作

            state = self._getDroneStateVector(k, self.need_target)  # 获取无人机状态并计算目标位置和偏航变化
            keep_pos = state['pos']
            keep_pos[2] = self.init_z  # 现在位置保持高度
            target_pos, yaw_change, safe_penalty[k] = self.map_action_to_target_position(
                action_id, state['pos'], state['rpy'][2]
            )
            if yaw_change != 0:
                # 计算目标RPM值
                rpm[k, :], _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state['pos'],
                    cur_quat=state['quat'],
                    cur_vel=state['vel'],
                    cur_ang_vel=state['ang_vel'],
                    target_pos=keep_pos,  # 使用位置移动时：target_pos,速度时：keep_pos  # 与当前位置相同
                    target_rpy=np.array([0, 0, state['rpy'][2] + 0.2 * yaw_change]),  # 添加目标偏航
                )
            else:
                rpm[k, :], _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state['pos'],
                    cur_quat=state['quat'],
                    cur_vel=state['vel'],
                    cur_ang_vel=state['ang_vel'],
                    target_pos=target_pos,  # 使用位置移动时：target_pos,速度时：keep_pos  # 与当前位置相同
                    target_rpy=np.array([0, 0, state['rpy'][2]]),  # 添加目标偏航
                )
        # print(rpm)
        return rpm, safe_penalty

    def enforce_position_limits(self, target_pos, current_pos):
        safe_penalty = 0
        # 限制x轴位置
        if target_pos[0] < -2:
            target_pos[0] = max(current_pos[0] - 0.5, -2)
            safe_penalty += 20
        elif target_pos[0] > 2:
            target_pos[0] = min(current_pos[0] + 0.5, 2)
            safe_penalty += 20
        # 限制y轴位置
        if target_pos[1] < -2:
            target_pos[1] = max(current_pos[1] - 0.5, -2)
            safe_penalty += 20
        elif target_pos[1] > 2:
            target_pos[1] = min(current_pos[1] + 0.5, 2)
            safe_penalty += 20
        return safe_penalty, target_pos

    def map_action_to_target_position(self, action_id, current_pos, current_yaw):
        cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        action_map = {
            0: (np.array([0.8, 0, 0]), 0),  # 全速前进
            1: (np.array([0.3, 0, 0]), 0),  # 缓速前进
            2: (np.array([-0.3, 0, 0]), 0),  # 后退
            3: (np.array([0, 0, 0]), 0),  # 保持位置
            4: (np.array([0, 0, 0]), -0.8),  # 全速左转
            5: (np.array([0, 0, 0]), -0.2),  # 缓速左转
            6: (np.array([0, 0, 0]), 0.8),  # 全速右转
            7: (np.array([0, 0, 0]), 0.2)  # 缓速右转
        }
        v_unit_vector, yaw_change = action_map.get(action_id, (np.array([0, 0, 0]), 0))
        v_unit_vector = rotation_matrix @ v_unit_vector
        target_pos = current_pos + v_unit_vector * self.CTRL_TIMESTEP * self.SPEED_LIMIT
        # 强制目标位置在安全范围内
        safe_penalty, target_pos = self.enforce_position_limits(target_pos, current_pos)
        return target_pos, yaw_change, safe_penalty

    def enforce_altitude_limits(self, pos, v_unit_vector):
        safe_penalty = 0
        if pos[0] < -2 and v_unit_vector[0] < 0:  # 限制x轴位置
            v_unit_vector[0] = 0.5
            safe_penalty += 10
        elif pos[0] > 2 and v_unit_vector[0] > 0:
            v_unit_vector[0] = -0.5
            safe_penalty += 10
        if pos[1] < -2 and v_unit_vector[1] < 0:  # 限制y轴位置
            v_unit_vector[1] = 0.5
            safe_penalty += 10
        elif pos[1] > 2 and v_unit_vector[1] > 0:
            v_unit_vector[1] = -0.5
            safe_penalty += 10
        if pos[2] < -2 and v_unit_vector[2] < 0:  # 限制y轴位置
            v_unit_vector[2] = 0.5
            safe_penalty += 10
        elif pos[2] > 2 and v_unit_vector[2] > 0:
            v_unit_vector[2] = -0.5
            safe_penalty += 10
        return safe_penalty, v_unit_vector

    def map_action_to_movement(self, action_id, current_yaw):
        cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)  # 创建旋转矩阵用于将速度向量从机体坐标系转换到世界坐标系
        rotation_matrix = np.array([        # 旋转矩阵，用于将无人机在当前航向上的前进后退方向转换为世界坐标系
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]])
        action_map = {
            0: (np.array([1, 0, 0]), 0),    # 全速前进
            1: (np.array([0.5, 0, 0]), 0),  # 缓速前进
            2: (np.array([-0.5, 0, 0]), 0),  # 后退
            3: (np.array([0, 0, 0]), 0),    # 保持位置
            4: (np.array([0, 0, 0]), -0.2),  # 全速左转
            5: (np.array([0, 0, 0]), -0.1),  # 缓速左转
            6: (np.array([0, 0, 0]), 0.2),  # 全速右转
            7: (np.array([0, 0, 0]), 0.1)}  # 缓速右转
        v_unit_vector, yaw_change = action_map.get(action_id, (np.array([0, 0, 0]), 0))     # 获取基础的速度向量和偏航角改变
        v_unit_vector = rotation_matrix @ v_unit_vector         # 将速度向量旋转到无人机当前航向的方向
        return v_unit_vector, yaw_change

    ################################################################################

    def _observationSpace(self, Obs_act=False):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            这是观测空间的定义，下面有观测的计算过程
            A Box() of shape [NUM_DRONES,(H,W,3)] or [NUM_DRONES,12] depending on the observation type.
        """
        if self.OBS_TYPE == ObservationType.RGB:
            space_1 = [spaces.Box(low=0, high=255,  # 4 for only rgba
                                  shape=(self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8) for _ in
                       range(self.NUM_DRONES)]
            lo, hi = -np.inf, np.inf
            # 创建 obs_bound,           X    Y   Z   R   P   Y   VX  VY  VZ  WX  WY  WZ
            obs_lower_bound = np.array([lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo])
            obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi])
            #### Add action buffer to observation space ################
            if self.discrete:
                space_2 = [spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32) for _ in
                           range(self.NUM_DRONES)]
                return space_1, space_2    # RBGs, states(12), （actions从动作观测中获取吧）
            else:
                if Obs_act == True:  # 6/12 观测动作为Flase 这样避免了麻烦
                    act_lo, act_hi = -1, +1  # 动作为VEL_YAW 5维
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo, act_lo, act_lo, act_lo, act_lo])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi, act_hi, act_hi, act_hi, act_hi])])
                space_2 = [spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32) for _ in
                           range(self.NUM_DRONES)]
                return space_1, space_2
        elif self.OBS_TYPE == ObservationType.KIN_target:  # 位姿加上目标位置+4维动作  #不需要+(num_drones-1)*其他无人机位置
            ############################################################
            lo, hi = -np.inf, np.inf
            # 创建 obs_bound,           X    Y   Z   R   P   Y   VX  VY  VZ  WX  WY  WZ  TX, TY, TZ, Tpos
            obs_lower_bound = np.array([lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, 0])
            obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi])
            # 使用 np.tile 扩展位置和距离边界
            position_bounds_lower = np.tile([lo, lo, lo, 0], self.NUM_DRONES - 1)
            position_bounds_upper = np.tile([hi, hi, hi, hi], self.NUM_DRONES - 1)
            # 连接初始边界和扩展的边界
            obs_lower_bound = np.concatenate((obs_lower_bound, position_bounds_lower))
            obs_upper_bound = np.concatenate((obs_upper_bound, position_bounds_upper))
            #### Add action buffer to observation space ################
            if Obs_act == True:  # 6/12 观测动作为Flase 这样避免了麻烦
                act_lo, act_hi = -1, +1
                # for i in range(self.ACTION_BUFFER_SIZE):  # 30//2 次   只保存一次的动作
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo, act_lo, act_lo, act_lo])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi, act_hi, act_hi, act_hi])])
                elif self.ACT_TYPE == ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo, act_lo, act_lo])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi, act_hi, act_hi])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi])])
                elif self.ACT_TYPE == ActionType.V_YAW:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([act_lo, act_lo, act_lo, act_lo, act_lo])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([act_hi, act_hi, act_hi, act_hi, act_hi])])
            return [spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32) for _ in
                    range(self.NUM_DRONES)]
            ############################################################
        else:
            print("[ERROR] LyyRLAviary._observationSpace()")

    ################################################################################
    def _computeAllObs(self, Obs_act=False):
        """Returns the current observation of the environment.
            这里需要注意修改后保证上面的观测空间一致
            如果观测有 target 则返回 dict
        Returns
        -------
        ndarray
            A RGB uint8 array, A Dict of obs
        """
        # if self.step_counter % self.IMG_CAPTURE_FREQ == 0:  # 应该是每240 step 拍一张，现在每1帧后等待29帧满足240steps.1s难以接受，直接使用30Hz吧
        for i in range(self.NUM_DRONES):
            self.rgb[i], _, _ = self._getDroneImages(i, segmentation=False)
        obs_dict = {}
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i, self.need_target)  # 如果True， obs['target_pos']是无人机指向目标的向量
            obs_dict[i] = {
                'pos': obs['pos'],  # 3
                'rpy': obs['rpy'],  # 3
                'vel': obs['vel'],  # 3
                'ang_vel': obs['ang_vel'],  # 3
                'last_action': self.action_buffer[-1][i]  # 添加一个动作 5
            }
        return self.rgb, self.to_array_obs(obs_dict)
        ############################################################

    def convert_obs_dict_to_array(self, obs_dict):
        obs_array = []
        for i in range(self.NUM_DRONES):
            obs = obs_dict[i]
            # action_buffer_flat = np.hstack(obs['action_buffer'])    # 拉成一维
            obs_array.append(np.hstack([
                obs['pos'],
                obs['rpy'],
                obs['vel'],
                obs['ang_vel'],
                obs['last_action']
            ]))
        return np.array(obs_array).astype('float32')

    def to_array_obs(self, obs_dict):
        if isinstance(obs_dict, dict):
            obs_array = self.convert_obs_dict_to_array(obs_dict)
        else:
            obs_array = obs_dict
        return obs_array

    def _computeObs(self, Obs_act=False):   # 现在没有使用
        """Returns the current observation of the environment.
            这里需要注意修改后保证上面的观测空间一致
            如果观测有 target 则返回 dict
        Returns
        -------
        ndarray
            A Dict of obs
        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH + "drone_" + str(i),
                                          frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                                          )
            return self.rgb
        ############################################################
        elif self.OBS_TYPE == ObservationType.KIN_target:  # 添加目标位置,其他智能体信息相当于通信信息而非观测
            obs_dict = {}
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i, self.need_target)  # 如果True， obs['target_pos']是无人机指向目标的向量
                if self.NUM_DRONES != 1:  # 有多架无人机时 有队友位置
                    obs_dict[i] = {
                        'pos': obs['pos'],  # 3
                        'rpy': obs['rpy'],  # 3
                        'vel': obs['vel'],  # 3
                        'ang_vel': obs['ang_vel'],  # 3
                        'target_pos': obs['target_pos_dis'],  # 4
                        # 'target_dis': obs['target_dis'],      # 1
                        'other_pos': obs['other_pos_dis'],  # 4*(N-1)
                        # 'other_dis': obs['other_dis'],        # N-1
                        'last_action': self.action_buffer[-1][i]  # 添加一个动作 4/3
                    }
                else:
                    obs_dict[i] = {
                        'pos': obs['pos'],
                        'rpy': obs['rpy'],
                        'vel': obs['vel'],
                        'ang_vel': obs['ang_vel'],
                        'target_pos': obs['target_pos'],
                        'target_dis': obs['target_dis'],
                        'action_buffer': obs['last_clipped_action']  # # 添加一个动作 4/3
                    }
            return obs_dict
            ############################################################
        else:
            print("[ERROR] in LyyRLAviary._computeObs()")
