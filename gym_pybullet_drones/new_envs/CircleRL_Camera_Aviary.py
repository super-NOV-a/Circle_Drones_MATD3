import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

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
                 test=False  # 若测试，则实例化对象时给定初始位置
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
        if self.OBS_TYPE == ObservationType.RGB:
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
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
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
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
        #
        return [spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32) for _ in range(self.NUM_DRONES)]

    ################################################################################

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differently for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))  # 最终计算结果为rpm
        for k in range(self.NUM_DRONES):  # 第 k 架drones
            target = action[k]  # 动作直接作为各个方法的目标
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k, :] = np.array(self.HOVER_RPM * (1 + 0.05 * target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k, self.need_target)
                next_pos = self._calculateNextStep(
                    current_position=state['pos'],
                    destination=target,
                    step_size=1,
                )
                rpm_k, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,  # 1/Ctrl_freq=0.001 s
                    cur_pos=state['pos'],
                    cur_quat=state['quat'],
                    cur_vel=state['vel'],
                    cur_ang_vel=state['ang_vel'],
                    target_pos=next_pos
                )
                rpm[k, :] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k, self.need_target)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state['pos'],
                    cur_quat=state['quat'],
                    cur_vel=state['vel'],
                    cur_ang_vel=state['ang_vel'],
                    target_pos=state['pos'],  # same as the current position
                    target_rpy=np.array([0, 0, state['rpy'][2]]),  # keep current yaw
                    target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector
                    # target the desired velocity vector
                )
                rpm[k, :] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k, :] = np.repeat(self.HOVER_RPM * (1 + 0.05 * target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state['pos'],
                    cur_quat=state['quat'],
                    cur_vel=state['vel'],
                    cur_ang_vel=state['ang_vel'],
                    target_pos=state['pos'] + 0.1 * np.array([0, 0, target[0]])
                )
                rpm[k, :] = res
            elif self.ACT_TYPE == ActionType.V_YAW:  # [0:4]维速度，第4维偏航角
                state = self._getDroneStateVector(k, self.need_target)  # get image
                if np.linalg.norm(target[:3]) != 0:
                    v_unit_vector = target[:3] / np.linalg.norm(target[:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state['pos'],
                    cur_quat=state['quat'],
                    cur_vel=state['vel'],
                    cur_ang_vel=state['ang_vel'],
                    target_pos=state['pos'],  # same as the current position
                    target_rpy=np.array([0, 0, state['rpy'][2] + target[4]*0.2]),  # add target yaw
                    target_vel=self.SPEED_LIMIT * np.abs(target[3]) * 0.4 * v_unit_vector  # 额外乘0.4保证飞行稳定
                )
                rpm[k, :] = temp
            else:
                print("[ERROR] _preprocessAction()")
                exit()
        return rpm

    def try_continue(self, action):  # 仅测试一个
        rpm = np.zeros((self.NUM_DRONES, 4))  # 最终计算结果为rpm
        target = action
        if self.ACT_TYPE == ActionType.VEL:
            state = self._getDroneStateVector(0, self.need_target)
            if np.linalg.norm(target[0:3]) != 0:
                v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
            else:
                v_unit_vector = np.zeros(3)
            temp, _, _ = self.ctrl[0].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state['pos'],
                cur_quat=state['quat'],
                cur_vel=state['vel'],
                cur_ang_vel=state['ang_vel'],
                target_pos=state['pos'],  # same as the current position
                target_rpy=np.array([0, 0, state['rpy'][2]]),  # keep current yaw
                target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector
                # target the desired velocity vector
            )
            rpm[0, :] = temp
        return rpm

    def try_discrete(self, action):
        rpm = np.zeros((self.NUM_DRONES, 4))
        state = self._getDroneStateVector(0, self.need_target)
        target = state['pos'].copy()  # 开始时目标就是原位置，.copy保证不会优化掉
        target_yaw = state['rpy'][2]
        dist = 0.1
        if action == 0:
            print('前进')
            target[0] += dist * np.cos(target_yaw)  # x方向移动，根据偏航角调整
            target[1] += dist * np.sin(target_yaw)  # y方向移动，根据偏航角调整
        elif action == 1:
            print('后退')
            target[0] -= dist * np.cos(target_yaw)  # x方向移动，根据偏航角调整
            target[1] -= dist * np.sin(target_yaw)  # y方向移动，根据偏航角调整
        elif action == 2:
            target_yaw += 0.5  # yaw增加
        elif action == 3:
            target_yaw -= 0.5  # yaw减小
        else:
            pass
        print('移动向量:', target - state['pos'])
        next_pos = self._calculateNextStep(
            current_position=state['pos'],
            destination=target,
            step_size=1,
        )
        rpm_k, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,  # 1/Ctrl_freq=0.001 s
            cur_pos=state['pos'],
            cur_quat=state['quat'],
            cur_vel=state['vel'],
            cur_ang_vel=state['ang_vel'],
            target_pos=next_pos,
            target_rpy=np.array([0, 0, target_yaw])
        )
        rpm[0, :] = rpm_k
        return rpm

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
            A RGB array, A Dict of obs
        """
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0:  # 应该是每240 step 拍一张
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
        return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32'), self.to_array_obs(obs_dict)
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

    def _computeObs(self, Obs_act=False):
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
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
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
