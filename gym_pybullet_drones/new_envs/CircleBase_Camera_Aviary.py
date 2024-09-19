import os
import random
import time
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from matplotlib import pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType


def generate_non_overlapping_positions_numpy(scale=1.0):
    """
    生成不重叠的位置，并根据 scale 参数调整生成范围的大小。

    参数:
    scale (float): 用于调整生成范围的大小。默认为 1，即生成范围为 [-1, 1]。

    返回:
    list: 生成的位置列表，每个位置为 (x, y, z) 的元组。
    """
    cell_size = 0.5    # 单元格大小固定为 0.5
    total_range = 2 * scale    # 计算新的总范围
    divisions = int(total_range / cell_size)    # 计算划分数目
    cell_coordinates = np.array([(x, y) for x in range(divisions) for y in range(divisions)])    # 生成所有可能的单元格坐标
    np.random.shuffle(cell_coordinates)
    z = np.random.uniform(0.1, 1)  # 在 [0, 1] 范围内生成随机 z 坐标，全部无人机共用
    positions = []  # 生成位置列表
    for cell_coord in cell_coordinates:    # 生成位置
        x = np.random.uniform(cell_coord[0] * cell_size - scale, (cell_coord[0] + 1) * cell_size - scale)
        y = np.random.uniform(cell_coord[1] * cell_size - scale, (cell_coord[1] + 1) * cell_size - scale)

        positions.append((x, y, z))
    return positions

# 例子
# positions = generate_non_overlapping_positions_numpy(scale=2)


class CircleBaseCameraAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results',
                 need_target=False,
                 obs_with_act=False,
                 test=False,  # 若测试，则实例化对象时给定初始位置
                 discrete=False
                 ):
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.MODEL_FREQ = 1  # todo 设计模型推理频率
        self.CTRL_FREQ = 30  # 控制器更新频率
        self.PYB_FREQ = 240  # default 240 物理更新频率
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)  # default 240/30=8
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ  # 飞控控制器 控制周期Tc    1/30
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ  # 仿真周期Tp
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        self.need_target = need_target  # added
        self.Test = test
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, self.L, self.THRUST2WEIGHT_RATIO, self.J, self.J_INV, self.KF, self.KM, self.COLLISION_H, \
            self.COLLISION_R, self.COLLISION_Z_OFFSET, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, \
            self.DRAG_COEFF, self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3 = self._parseURDFParameters()
        print(
            "[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
                self.M, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2],
                self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G * self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM ** 2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM ** 2) / np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM ** 2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM ** 2) / np.sqrt(2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM ** 2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM ** 2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        if self.RECORD:
            self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER,
                                                 "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([128, 96])
            self.IMG_FRAME_PER_SEC = 1
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ / self.IMG_FRAME_PER_SEC)  # 周期为：240/1 步
            self.rgb = np.zeros((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ % self.PYB_STEPS_PER_CTRL != 0:  # 8/8
                print(
                    "[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(
                        self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                for i in range(self.NUM_DRONES):
                    os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/"), exist_ok=True)
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI)  # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                      p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1 * np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller " + str(i) + " RPM", 0, self.MAX_RPM,
                                                              self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH = int(640)
                self.VID_HEIGHT = int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.PYB_FREQ / self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH / self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyzs is None:  # todo 修改初始位置
            # 正负范围：0.8:9个随机cell位置，1.0: 16个，1.3: 25个，1.5: 36个,1.8: 49个,2.0: 64个
            self.cell_pos = generate_non_overlapping_positions_numpy(1)
            self.need_target = need_target  # 若需要，同时给出目标位置
            if self.need_target:
                self.INIT_XYZS, self.TARGET_POS = self.get_init()
            else:
                self.INIT_XYZS = self.get_init()
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES, 3):
            self.INIT_XYZS = initial_xyzs
            self.TARGET_POS = np.array([0, 0, 0.8])
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        if self.need_target:
            self.show_target()
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace(obs_with_act)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        self.detected_num = [0 for _ in range(self.NUM_DRONES)]

    ################################################################################
    def get_init(self):
        """
        :return: 若需要目标，则返回 无人机+目标 初始位置 init_pos[:3], init_pos[3]/target
        """
        if self.need_target:
            init_pos = np.stack(random.sample(self.cell_pos, self.NUM_DRONES + 1))
            return init_pos[:self.NUM_DRONES], init_pos[self.NUM_DRONES]
        else:
            init_pos = np.stack(random.sample(self.cell_pos, self.NUM_DRONES))
            # init_pos = np.array([[1, 1, 1], [-1, -1, 0], [1, -1, 1]])
            return init_pos

    def show_target(self):
        self.target_id = p.loadURDF("E:\PyProjects\MARL-code-pytorch\gym_pybullet_drones\\assets\cf2p.urdf",
                                    self.TARGET_POS,
                                    p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.CLIENT,
                                    useFixedBase=True)
        # 设置模型无重力
        p.changeDynamics(self.target_id, -1, mass=0)  # 将质量设为零，即设置无重力
        p.changeDynamics(self.target_id, -1, linearDamping=0, angularDamping=0)  # 设置运动类型为静态
        # 设置模型不受其他碰撞影响
        p.setCollisionFilterGroupMask(self.target_id, -1, 0, 0)  # 设置碰撞组掩码
        p.setCollisionFilterPair(-1, -1, self.target_id, -1, 0)  # 设置碰撞对，使模型不与其他对象发生碰撞

    def update_target_pos(self):
        """
        更新self.Target_pos 还需要加上self.relative_pos 才得到真实的目标pos
        :return:
        """
        pass
        # # 随着计数次数增加，目标位置的变化变大
        # gradient = np.arctan(self.step_counter) / 100  # step_counter~[0,1000000], gradient~[-0.01,0.01]
        # # 随机选择方向并归一化
        # direction = np.random.uniform(-1, 1, size=(3,))
        # direction /= np.linalg.norm(direction)
        #
        # # 根据变化程度和方向更新目标位置
        # self.TARGET_POS += gradient * direction
        # # 剪切到所需的范围内  target_pos 维度大小只有[1, 3]
        # self.TARGET_POS[:, 0:2] = np.clip(self.TARGET_POS[:, 0:2], -0.8, 0.8)
        # self.TARGET_POS[:, 2] = np.clip(self.TARGET_POS[:, 2], 0.2, 0.8)
        # # print("现在的目标位置修改为：", self.TARGET_POS)
        # p.resetBasePositionAndOrientation(self.target_id, self.TARGET_POS[0], p.getQuaternionFromEuler([0, 0, 0]))

    def reset(self,
              seed: int = None,
              options: dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        ####  给定目标位置在 _housekeeping中 ###########################################
        if self.need_target:
            # self.TARGET_POS = self.get_init_target()
            self.show_target()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_rgbs, initial_states = self._computeAllObs()
        initial_info = self._computeInfo()
        # self.see_ball()
        return initial_rgbs, initial_states, initial_info

    ################################################################################

    def step(self, action):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.
        交互step频率是30Hz包含8个物理仿真step，结束时增加8个step_counter,每秒图像观测一张        """
        clip_action, safe_penalty = self._preprocessAction(action, self.detected_num)
        clipped_action = np.reshape(clip_action, (self.NUM_DRONES, 4))

        for STEP in range(self.PYB_STEPS_PER_CTRL):     # 8步
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG,
                                                                Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            for i in range(self.NUM_DRONES):
                self.apply_physics(clipped_action[i, :], i)
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
        self._updateAndStoreKinematicInformation()
        self.last_clipped_action = clipped_action
        if self.need_target:
            self.update_target_pos()
        rgbs, states = self._computeAllObs()  # array, array
        rewards = self._computeReward() - safe_penalty
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        self.step_counter += (1 * self.PYB_STEPS_PER_CTRL)

        return rgbs, states, rewards, terminated, truncated, info

    ################################################################################

    def apply_physics(self, clipped_action, i):
        if self.PHYSICS == Physics.PYB:
            self._physics(clipped_action, i)
        elif self.PHYSICS == Physics.DYN:
            self._dynamics(clipped_action, i)
        elif self.PHYSICS == Physics.PYB_GND:
            self._physics(clipped_action, i)
            self._groundEffect(clipped_action, i)
        elif self.PHYSICS == Physics.PYB_DRAG:
            self._physics(clipped_action, i)
            self._drag(self.last_clipped_action[i, :], i)
        elif self.PHYSICS == Physics.PYB_DW:
            self._physics(clipped_action, i)
            self._downwash(i)
        elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
            self._physics(clipped_action, i)
            self._groundEffect(clipped_action, i)
            self._drag(self.last_clipped_action[i, :], i)
            self._downwash(i)

    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        if self.first_render_call and not self.GUI:
            print(
                "[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time() - self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter * self.PYB_TIMESTEP, self.PYB_FREQ,
                                                                (self.step_counter * self.PYB_TIMESTEP) / (
                                                                        time.time() - self.RESET_TIME)))
        for i in range(self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0] * self.RAD2DEG,
                                                                              self.rpy[i, 1] * self.RAD2DEG,
                                                                              self.rpy[i, 2] * self.RAD2DEG),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1],
                                                                                     self.ang_v[i, 2]))

    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT

    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS

    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)  # 用于设置调用stepSimulation时的步长
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)  # 用于增加导入模型的路径
        #### Load ground plane, drone and obstacles models #########
        if not self.Test:
            if self.need_target:
                self.INIT_XYZS, self.TARGET_POS = self.get_init()  # 重新给出位置
            else:
                self.INIT_XYZS = self.get_init()
        self.init_z = self.INIT_XYZS[0][2]
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_IDS = np.array(
            [p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF),
                        self.INIT_XYZS[i, :],
                        p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                        flags=p.URDF_USE_INERTIA_FROM_FILE,
                        physicsClientId=self.CLIENT
                        ) for i in range(self.NUM_DRONES)])
        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        for i in range(self.NUM_DRONES):
            p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1,
                                     linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()

    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range(self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)

    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.join(self.OUTPUT_FOLDER,
                                                                      "video-" + datetime.now().strftime(
                                                                          "%m.%d.%Y_%H.%M.%S") + ".mp4"),
                                                physicsClientId=self.CLIENT
                                                )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER,
                                         "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"), '')
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    ################################################################################

    def _getDroneStateVector(self, nth_drone, with_target=False):
        """Returns the state vector of the n-th drone. todo 修改环境 这里总是要检查的

            (3,   4,    3,   3,    3,       4*n,            4*(n-1),         4)

            (pos, quat, rpy, vel, ang_vel, target_pos_dis, other_pos_dis, last_clipped_action)
        """
        if with_target:
            state_dict = {
                'pos': self.pos[nth_drone, :],  # 3
                'quat': self.quat[nth_drone, :],  # 4
                'rpy': self.rpy[nth_drone, :],  # 3
                'vel': self.vel[nth_drone, :],  # 3
                'ang_vel': self.ang_v[nth_drone, :],  # 3
                'target_pos_dis': np.append(self.TARGET_POS[:] - self.pos[nth_drone, :],
                                            np.linalg.norm(self.TARGET_POS[:] - self.pos[nth_drone, :]))  # 4
            }
            other_pos_dis = []  # 存储智能体指向其他智能体的向量和距离 4*(N-1)
            for i in range(self.NUM_DRONES):
                if i != nth_drone:
                    pos = self.pos[i, :] - self.pos[nth_drone, :]
                    dis = np.linalg.norm(self.pos[i, :] - self.pos[nth_drone, :])
                    other_pos_dis.append(np.append(pos, dis))
            state_dict['other_pos_dis'] = np.array(other_pos_dis).flatten()  # 合并后的向量和距离
            # state_dict['last_clipped_action'] = self.last_clipped_action[nth_drone, :]  # 动作在RL文件中读取的
            return state_dict

        else:  # 不需要目标位置
            state_dict = {
                'pos': self.pos[nth_drone, :],  # 3
                'quat': self.quat[nth_drone, :],  # 4
                'rpy': self.rpy[nth_drone, :],  # 3
                'vel': self.vel[nth_drone, :],  # 3
                'ang_vel': self.ang_v[nth_drone, :],  # 3
            }
            other_pos_dis = []  # 存储智能体指向其他智能体的向量和距离 4*(N-1)
            for i in range(self.NUM_DRONES):
                if i != nth_drone:
                    pos = self.pos[i, :] - self.pos[nth_drone, :]
                    dis = np.linalg.norm(self.pos[i, :] - self.pos[nth_drone, :])
                    other_pos_dis.append(np.append(pos, dis))
            state_dict['other_pos_dis'] = np.array(other_pos_dis).flatten()  # 合并后的向量和距离
            # state_dict['last_clipped_action'] = self.last_clipped_action[nth_drone, :]  # 动作在RL文件中读取的
            return state_dict

    ################################################################################

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        """
        返回第n个无人机视角的相机捕获图像。

        参数
        ----------
        nth_drone : int   无人机在列表 self.DRONE_IDS 中的位置编号。
        segmentation : bool, 可选   是否计算分割掩码。  这会影响性能。

        返回
        -------
        ndarray  形状为 (h, w, 4) 的 uint8 类型数组，包含从第n个无人机视角捕获的 RGB(A) 图像。
        ndarray  形状为 (h, w) 的 uint8 类型数组，包含从第n个无人机视角捕获的深度图像。
        ndarray  形状为 (h, w) 的 uint8 类型数组，包含从第n个无人机视角捕获的分割图像。
        """
        if self.IMG_RES is None:
            print("[ERROR] 在 BaseAviary._getDroneImages() 中，记得将 self.IMG_RES 设置为 np.array([宽度, 高度])")
            exit()
        # 计算旋转矩阵
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        # 设置目标点，相机视角矩阵和投影矩阵 # 目标点为1000，表示正前方
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT)

        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                     aspect=1.0,
                                                     nearVal=self.L,
                                                     farVal=1000.0)

        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        # 获取相机图像
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT)

        rgb = np.reshape(rgb, (h, w, 4))  # RGBA,透明度(Alpha)
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))

        return rgb, dep, seg

    ################################################################################

    def _exportImage(self,
                     img_type: ImageType,
                     img_input,
                     path: str,
                     frame_num: int = 0
                     ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'), 'RGBA')).save(
                os.path.join(path, "frame_" + str(frame_num) + ".png"))
        elif img_type == ImageType.DEP:
            temp = ((img_input - np.min(img_input)) * 255 / (np.max(img_input) - np.min(img_input))).astype('uint8')
        elif img_type == ImageType.SEG:
            temp = ((img_input - np.min(img_input)) * 255 / (np.max(img_input) - np.min(img_input))).astype('uint8')
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype('uint8')
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(os.path.join(path, "frame_" + str(frame_num) + ".png"))

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES - 1):
            for j in range(self.NUM_DRONES - i - 1):
                if np.linalg.norm(self.pos[i, :] - self.pos[j + i + 1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j + i + 1] = adjacency_mat[j + i + 1, i] = 1
        return adjacency_mat

    ################################################################################

    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm ** 2) * self.KF
        torques = np.array(rpm ** 2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )

    ################################################################################

    def _groundEffect(self,
                      rpm,
                      nth_drone
                      ):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = p.getLinkStates(self.DRONE_IDS[nth_drone],
                                      linkIndices=[0, 1, 2, 3, 4],
                                      computeLinkVelocity=1,
                                      computeForwardKinematics=1,
                                      physicsClientId=self.CLIENT
                                      )
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array(
            [link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm ** 2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS / (4 * prop_heights)) ** 2
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and np.abs(self.rpy[nth_drone, 1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    ################################################################################

    def _drag(self,
              rpm,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot.T, drag_factors * np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )

    ################################################################################

    def _downwash(self,
                  nth_drone
                  ):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z)) ** 2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5 * (delta_xy / beta) ** 2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    ################################################################################

    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm ** 2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm ** 2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            z_torques = -z_torques
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          quat,
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            np.dot(rotation, rpy_rates),
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone, :] = rpy_rates

    def _integrateQ(self, quat, omega, dt):
        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([
            [0, r, -q, p],
            [-r, 0, p, q],
            [q, -p, 0, r],
            [-p, -q, -r, 0]
        ]) * .5
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

    ################################################################################

    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action + 1) * self.HOVER_RPM, self.HOVER_RPM + (
                self.MAX_RPM - self.HOVER_RPM) * action)  # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`

    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        p.loadURDF("samurai.urdf",
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("duck_vhacd.urdf",
                   [-.5, -.5, .05],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("cube_no_rotation.urdf",
                   [-.5, -2.5, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("sphere2.urdf",
                   [0, 2, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )

    ################################################################################

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _observationSpace(self, Obs_act=False):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    def _computeAllObs(self):
        raise NotImplementedError

    ################################################################################

    def _preprocessAction(self, action, detected_num):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs."""
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.        """
        raise NotImplementedError

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint
        towards drone's destination
        from drone's current position

        Enables drones to reach distant waypoints without
        losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int
            intermediate waypoint for drone

        """
        direction = (
                destination - current_position
        )  # Calculate the direction vector
        distance = np.linalg.norm(
            direction
        )  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = (
                direction / distance
        )  # Normalize the direction vector
        next_step = (
                current_position + normalized_direction * step_size
        )  # Calculate the next step
        return next_step
