import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.new_envs.CircleBaseAviary import CircleBaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class CircleRLAviary(CircleBaseAviary):
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
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.MIXED]:
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
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL or act == ActionType.MIXED:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides LyyBaseAviary's method.

        """
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
        list of spaces.Tuple
            A list of Tuples, each containing a continuous and a discrete action space for each drone.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        elif self.ACT_TYPE == ActionType.MIXED:  # 新增的混合动作类型
            cont_size = 3  # 三维连续动作
            # discrete_size = 1  # 默认一维离散动作
        else:
            print("[ERROR] in _actionSpace()")
            exit()

        if self.ACT_TYPE == ActionType.MIXED:
            cont_act_lower_bound = np.array(-1 * np.ones(cont_size))
            cont_act_upper_bound = np.array(+1 * np.ones(cont_size))
            discrete_act_space = spaces.Discrete(2)  # 离散动作空间：0 或 1

            action_space = spaces.Tuple((
                spaces.Box(low=cont_act_lower_bound, high=cont_act_upper_bound, dtype=np.float32),
                discrete_act_space
            ))

            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append((np.zeros((self.NUM_DRONES, cont_size)), np.zeros(self.NUM_DRONES)))

            return [action_space for _ in range(self.NUM_DRONES)]
        else:
            act_lower_bound = np.array(-1 * np.ones(size))
            act_upper_bound = np.array(+1 * np.ones(size))

            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))

            return [spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32) for _ in
                    range(self.NUM_DRONES)]

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
                state = self._getDroneStateVector(k)
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
                state = self._getDroneStateVector(k, True)
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
            elif self.ACT_TYPE == ActionType.MIXED:
                cont_action = action[k][0]
                discrete_action = action[k][1]

                target = cont_action  # 连续动作的目标
                if discrete_action == 0:    # 处理离散动作部分
                    target = np.zeros_like(target)  # 保持静止

                state = self._getDroneStateVector(k, True)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
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
            else:
                print("[ERROR] _preprocessAction()")
                exit()
        return rpm

    def try_mixed(self, action):    # 仅测试一个
        rpm = np.zeros((self.NUM_DRONES, 4))  # 最终计算结果为rpm

        if self.ACT_TYPE == ActionType.MIXED:
            cont_action = action[0]
            discrete_action = action[1]
            target = cont_action  # 连续动作的目标
            # 处理离散动作部分
            if discrete_action == 0:
                # 保持静止
                target = np.zeros_like(target)

            state = self._getDroneStateVector(0, True)
            if np.linalg.norm(target[0:3]) != 0:
                v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
            else:
                v_unit_vector = np.zeros(3)
            temp, _, _ = self.ctrl[0].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                     cur_pos=state['pos'],
                                                     cur_quat=state['quat'],
                                                     cur_vel=state['vel'],
                                                     cur_ang_vel=state['ang_vel'],
                                                     target_pos=state['pos'],  # same as the current position
                                                     target_rpy=np.array([0, 0, state['rpy'][2]]),
                                                     # keep current yaw
                                                     target_vel=self.SPEED_LIMIT * np.linalg.norm(target) * v_unit_vector
                                                     # target the desired velocity vector
                                                     )
            rpm[0, :] = temp
        return rpm
    ################################################################################

    def _observationSpace(self, Obs_act=False):
        """Returns the observation space of the environment.
        Returns
        -------
        ndarray
            这是观测空间的定义，下面有观测的计算过程
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.
        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN_target:  # 位姿加上目标位置+4维动作  #不需要+(num_drones-1)*其他无人机位置
            ############################################################
            lo = -np.inf
            hi = np.inf
            # 创建 obs_bound,           X    Y   Z   R   P   Y   VX  VY  VZ  WX  WY  WZ  TX, TY, TZ, Tpos
            obs_lower_bound = np.array([lo, lo, lo,  lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, 0])
            obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi])
            # 使用 np.tile 扩展位置和距离边界
            position_bounds_lower = np.tile([lo, lo, lo, 0], self.NUM_DRONES - 1)
            position_bounds_upper = np.tile([hi, hi, hi, hi], self.NUM_DRONES - 1)
            # 连接初始边界和扩展的边界
            obs_lower_bound = np.concatenate((obs_lower_bound, position_bounds_lower))
            obs_upper_bound = np.concatenate((obs_upper_bound, position_bounds_upper))
            #### Add action buffer to observation space ################
            if Obs_act == True:     # 6/12 观测动作为Flase 这样避免了麻烦
                act_lo = -1
                act_hi = +1
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
            # print(obs_upper_bound)
            return [spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32) for _ in
                    range(self.NUM_DRONES)]
            ############################################################
        else:
            print("[ERROR] LyyRLAviary._observationSpace()")

    ################################################################################

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
        elif self.OBS_TYPE == ObservationType.KIN:
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES, 12))
            for i in range(self.NUM_DRONES):
                # obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            return ret
        ############################################################
        elif self.OBS_TYPE == ObservationType.KIN_target:  # 添加目标位置,其他智能体信息相当于通信信息而非观测
            obs_dict = {}
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i, True)  # 如果True， obs['target_pos']是无人机指向目标的向量
                if self.NUM_DRONES != 1:    # 有多架无人机时 有队友位置
                    obs_dict[i] = {
                        'pos': obs['pos'],      # 3
                        'rpy': obs['rpy'],      # 3
                        'vel': obs['vel'],      # 3
                        'ang_vel': obs['ang_vel'],              # 3
                        'target_pos': obs['target_pos_dis'],    # 4
                        # 'target_dis': obs['target_dis'],      # 1
                        'other_pos': obs['other_pos_dis'],      # 4*(N-1)
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
