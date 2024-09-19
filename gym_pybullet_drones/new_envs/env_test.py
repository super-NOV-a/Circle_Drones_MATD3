import numpy as np
from gym_pybullet_drones.new_envs.CircleSpread_Camera import CircleCameraAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import matplotlib.pyplot as plt

# 初始化环境
Env_name = 'circle'  # 'spread3d', 'simple_spread'
action = 'vel_yaw'
NUM_DRONES = 3  # 无人机数量
test = True
xyzs = np.array([[-1, -1, 0.8], [1, 0, 0.8], [0, 1, 0.8]]) if test else None
need_target = True
env = CircleCameraAviary(num_drones=NUM_DRONES, initial_xyzs=xyzs, gui=True, obs=ObservationType('rgb'),
                         act=ActionType(action), need_target=need_target, obs_with_act=True, test=test, discrete=True)

# 初始目标速度和偏航角度
actions = np.array([0 for _ in range(NUM_DRONES)])  # x, y, z 方向速度以及速度大小


def get_drone_state(env, drone_id=0):
    state = env._getDroneStateVector(drone_id, env.need_target)
    pos = state['pos']
    quat = state['quat']
    vel = state['vel']
    ang_vel = state['ang_vel']
    rpy = state['rpy']
    return np.array(pos), quat, np.array(vel), np.array(ang_vel), np.array(rpy)


def detect_yellow_red_regions(image):
    yellow_min = np.array([180, 180, 0], dtype=np.uint8)
    yellow_max = np.array([255, 255, 70], dtype=np.uint8)
    red_min = np.array([180, 0, 0], dtype=np.uint8)
    red_max = np.array([255, 70, 70], dtype=np.uint8)

    img_rgb = image[:, :, :3]

    yellow_mask = np.all((img_rgb >= yellow_min) & (img_rgb <= yellow_max), axis=-1)
    green_mask = np.all((img_rgb >= red_min) & (img_rgb <= red_max), axis=-1)
    return yellow_mask, green_mask


def show_figure(fig, axes, rgb, yellow_mask, red_mask):
    axes[0].imshow(rgb[:, :, 0], cmap='gray')
    axes[0].set_title("R Image")
    axes[0].axis('off')
    axes[1].imshow(rgb[:, :, 1], cmap='gray')
    axes[1].set_title("G Image")
    axes[1].axis('off')
    axes[2].imshow(rgb[:, :, 2], cmap='gray')
    axes[2].set_title("B Image")
    axes[2].axis('off')

    # 创建一个全白色的图像
    Y = np.ones((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8) * 255
    Y[yellow_mask] = [255, 255, 0]
    Y[red_mask] = [255, 0, 0]
    axes[3].imshow(Y)
    axes[3].set_title("Detection Image")
    axes[3].axis('off')
    fig.canvas.flush_events()  # 处理图形事件


# 设置图像显示
plt.ion()  # 开启交互模式
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
plt.show(block=False)  # 不阻塞地显示窗口

rgb, dep, seg = env._getDroneImages(nth_drone=0, segmentation=False)
done = False
input('回车启动,输入速度方向和大小以及偏航角度增量')
while not done:
    # rgb, dep, seg = env._getDroneImages(nth_drone=0, segmentation=True)  # rgb: 96,128,4
    # yellow_mask, red_mask = detect_yellow_red_regions(rgb)
    # show_figure(fig, axes, rgb, yellow_mask, red_mask)
    action_input = input('3个智能体的动作:（分别是0-7动作的概率分布）')  # todo 修改为三个无人机动作的概率分布
    # 0.13243707 0.12140542 0.12257885 0.118792 0.1231009 0.12414426 0.1308618 0.1266796 0.13333048 0.11851338 0.12312149 0.12077631 0.12801889 0.12291928 0.12531583 0.12800433 0.13425545 0.12110135 0.12147482 0.11911386 0.12387094 0.12306637 0.12980565 0.12731153
    # 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1

    if action_input == 'r':
        env.reset()
        continue
    if action_input == 'e' or action_input == 'exit':
        break
    elif action_input.strip() != '':
        inputs = list(map(float, action_input.split()))
        if len(inputs) == 24:
            actions = np.array(inputs).reshape((3, 8))
    rgbs, _, rewards, _, _, _ = env.step(actions)
    rgb = rgbs[0]
    print('奖励值', rewards)

plt.ioff()  # 关闭交互模式
plt.close()
env.close()
