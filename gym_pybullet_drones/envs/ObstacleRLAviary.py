import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.ObstacleBaseAviary import ObstacleBaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class ObstacleRLAviary(ObstacleBaseAviary):
    """Lyy Base single and multi-agent environment class for reinforcement learning.
        Note : 无人机最优的位置应该是一个环上!!!
    """
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
                 ctrl_freq: int = 240,
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
        self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)  # 这里动作维数并不是简单的 N ，而是有之前动作的缓存 k*N
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
                         num_obstacle=num_obstacle,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
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

            for i in range(self.ACTION_BUFFER_SIZE):    # 初始时填充为0
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
            elif self.ACT_TYPE == ActionType.VEL:
                # state = self._getDroneStateVector(k, True)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=self.pos[k],
                    cur_quat=self.quat[k],
                    cur_vel=self.vel[k],
                    cur_ang_vel=self.ang_v[k],
                    target_pos=self.pos[k],  # same as the current position
                    target_rpy=np.array([0, 0, self.rpy[k][2]]),  # keep current yaw
                    target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector * 0.8
                    # target the desired velocity vector
                )
                rpm[k, :] = temp
            else:
                print("[ERROR] _preprocessAction()")
                exit()
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
        lo, hi, act_lo, act_hi = -np.inf, np.inf, -1, +1
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            obs_lower_bound = np.array(
                [[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array(
                [[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack(
                        [obs_lower_bound, np.array([[act_lo, act_lo, act_lo, act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack(
                        [obs_upper_bound, np.array([[act_hi, act_hi, act_hi, act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE == ActionType.PID:
                    obs_lower_bound = np.hstack(
                        [obs_lower_bound, np.array([[act_lo, act_lo, act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack(
                        [obs_upper_bound, np.array([[act_hi, act_hi, act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        elif self.OBS_TYPE == ObservationType.KIN_target_PO:  # 位姿加上目标位置+3维势能+4维动作  #需要+(num_drones-1)*其他无人机位置
            ############################################################
            lo = -np.inf
            hi = np.inf
            # 创建 obs_bound,           X    Y   Z   R   P   Y   VX  VY  VZ  WX  WY  WZ  TX, TY, TZ, Tpos
            obs_lower_bound = np.array([lo, lo, lo,  lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, 0])
            obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi])
            # 使用 np.tile 扩展位置和距离边界  POS for others
            position_bounds_lower = np.tile([lo, lo, lo, 0], self.NUM_DRONES - 1)
            position_bounds_upper = np.tile([hi, hi, hi, hi], self.NUM_DRONES - 1)
            # 新增势能作为obs    三维方向三维的力
            Fs_lower = np.array([lo, lo, lo])
            Fs_upper = np.array([hi, hi, hi])
            # 连接初始边界和扩展的边界
            obs_lower_bound = np.concatenate((obs_lower_bound, position_bounds_lower, Fs_lower))
            obs_upper_bound = np.concatenate((obs_upper_bound, position_bounds_upper, Fs_upper))
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
        elif self.OBS_TYPE == ObservationType.KIN_target:  # 位姿加上目标位置+4维动作  #需要+(num_drones-1)*其他无人机位置
            ############################################################
            # 创建 obs_bound,           X    Y   Z   R   P   Y   VX  VY  VZ  WX  WY  WZ  TX, TY, TZ, Tpos
            obs_lower_bound = np.array([lo, lo, lo,  lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, lo, 0])
            obs_upper_bound = np.array([hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi])
            # 使用 np.tile 扩展位置和距离边界  POS for others
            position_bounds_lower = np.tile([lo, lo, lo, 0], self.NUM_DRONES - 1)
            position_bounds_upper = np.tile([hi, hi, hi, hi], self.NUM_DRONES - 1)
            # 连接初始边界和扩展的边界
            obs_lower_bound = np.concatenate((obs_lower_bound, position_bounds_lower))
            obs_upper_bound = np.concatenate((obs_upper_bound, position_bounds_upper))
            #### Add action buffer to observation space ################
            if Obs_act == True:     # 6/12 观测动作为Flase 这样避免了麻烦
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
        elif self.OBS_TYPE == ObservationType.KIN_target:  # 添加目标位置,其他智能体信息相当于通信信息而非观测
            obs_dict = {}
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i, True)  # 如果True， obs['target_pos']是无人机指向目标的向量
                if self.NUM_DRONES != 1:  # 有多架无人机时 有队友位置
                    obs_dict[i] = {
                        'pos': obs['pos'],  # 3
                        'rpy': obs['rpy'],  # 3
                        'vel': obs['vel'],  # 3
                        'ang_vel': obs['ang_vel'],  # 3
                        'target_pos': obs['target_pos_dis'],  # 4
                        'other_pos': obs['other_pos_dis'],  # 4*(N-1)
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
                        'action_buffer': obs['last_clipped_action']  # 添加一个动作 4/3
                    }
            return obs_dict, False
            ############################################################
        else:
            print("[ERROR] in LyyRLAviary._computeObs()")


def potential_energy(obs_dict, num_agents, eta_att=1, eta_rep_agent=0.5, d0=1.0):
    """
    计算势能F，用于帮助critic收敛

    Parameters
    ----------
    obs_dict : dict
        每个无人机的观测字典，包含pos, rpy, vel, ang_vel, target_pos, other_pos, last_action
    num_agents : int
        总的无人机数量
    eta_att : float
        引力增益系数
    eta_rep_agent : float
        斥力增益系数
    d0 : float
        斥力感应范围
    n : int
        调节因子

    Returns
    -------
    F : np.array
        计算得到的势能向量 [fx, fy, fz]
    """
    # 计算引力F_att
    delta_lm = obs_dict['target_pos_dis'][:3]       # [3] 提取目标的相对位置
    dist_lm = obs_dict['target_pos_dis'][3]         # 提取目标的距离
    if dist_lm > 0:
        unit_lm = delta_lm / dist_lm            # 引力单位方向
        F_att_abs = eta_att / (dist_lm ** 2)    # 根据需求调整
        F_att = unit_lm * F_att_abs
    else:
        F_att = np.zeros(3)

    # 计算斥力F_rep_agent
    F_rep_agent = np.zeros(3)
    other_pos = obs_dict['other_pos_dis'].reshape((num_agents - 1, 4))
    for i in range(num_agents-1):
        delta_ag = other_pos[i][:3]     # [3] 提取其他无人机的相对位置
        dist_ag = other_pos[i][3]       # 提取其他无人机的距离
        if 0 < dist_ag < d0:                        # 感应斥力的范围默认是(0,1)
            unit_ag = delta_ag / dist_ag            # 斥力单位方向
            # 斥力1
            F_rep_ob1_abs = eta_rep_agent * (1/dist_ag - 1/d0) / (dist_ag ** 2)
            F_rep_ob1 = unit_ag * F_rep_ob1_abs
            # 斥力2（假设没有landmark，可以省略）
            # 如果有其他斥力来源，可以在这里添加
            F_rep_agent += F_rep_ob1
    # 总势能F
    F = F_att - F_rep_agent
    # 可选：将F缩放到某个范围内
    norm_F = np.linalg.norm(F)
    if norm_F > 0:
        F = F / norm_F  # 归一化
    else:
        F = np.zeros(3)
    return F
