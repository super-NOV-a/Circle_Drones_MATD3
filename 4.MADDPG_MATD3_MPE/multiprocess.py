import numpy as np
from matplotlib import pyplot as plt


def get_target(NUM_DRONES):
    # 初始化目标位置都在单位圆上
    angle_step = 2 * np.pi / NUM_DRONES
    target_pos = np.zeros((1, 3))
    relative_pos = np.zeros((NUM_DRONES, 3))
    target_pos[0, 2] = 0.8
    for i in range(NUM_DRONES):
        angle = i * angle_step
        relative_pos[i, 0] = 0.3 * np.cos(angle)
        relative_pos[i, 1] = 0.3 * np.sin(angle)
    # target_pos += relative_pos
    return target_pos, relative_pos


def update_target_pos(step_counter, TARGET_POS, relative_pos):
    pass
    # 随着计数次数增加，目标位置的变化变大
    gradient = np.arctan(step_counter) / 100  # step_counter~[0,1000000], gradient~[-0.01,0.01]
    # 随机选择方向并归一化
    direction = np.random.uniform(-1, 1, size=(3,))
    direction /= np.linalg.norm(direction)

    # 根据变化程度和方向更新目标位置
    TARGET_POS += gradient * direction
    # 剪切到所需的范围内
    TARGET_POS[:, 0:2] = np.clip(TARGET_POS[:, 0:2], -0.8, 0.8)
    TARGET_POS[:, 2] = np.clip(TARGET_POS[:, 2], 0.2, 1)
    return TARGET_POS


if __name__ == '__main__':
    target, relative = get_target(4)
    for i in range(1):
        target = update_target_pos(i, target, relative)
        print(target)
    points = target + relative
    print(target, relative, points)

    # 创建一个3D绘图对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制目标点
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='r', marker='o', label='Target')

    # 绘制相对位置点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='^', label='Relative Points')

    # 设置图例
    ax.legend()

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置标题
    ax.set_title('Target and Relative Points')

    # 显示图形
    plt.show()
