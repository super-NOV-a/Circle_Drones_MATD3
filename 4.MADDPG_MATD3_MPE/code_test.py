import numpy as np
import matplotlib.pyplot as plt


def get_arc_point(step, total_steps, start_point, end_point):
    """
    获取圆弧上的第 step 个点的三维坐标

    参数:
    step: 当前步骤，从 0 到 total_steps
    total_steps: 总步骤数
    start_point: 起点坐标 (x, y, z)
    end_point: 终点坐标 (x, y, z)

    返回:
    三维坐标 (x, y, z)
    """
    # 计算圆心和半径
    center = (start_point[:2] + end_point[:2]) / 2
    radius = np.linalg.norm(start_point[:2] - end_point[:2]) / 2

    # 计算角度范围，从 0 到 π
    theta = np.pi * (step / total_steps)

    # 计算圆弧上的 x 和 y 坐标
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # 计算旋转角度
    angle = np.arctan2(start_point[1] - end_point[1], start_point[0] - end_point[0])

    # 旋转矩阵
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # 旋转并平移圆弧上的点
    arc_point_2d = np.dot(rotation_matrix, np.array([x, y]))
    arc_point_2d[0] += center[0]
    arc_point_2d[1] += center[1]

    # 线性插值 z 坐标
    z = np.linspace(start_point[2], end_point[2], total_steps + 1)[step]

    return np.array([arc_point_2d[0], arc_point_2d[1], z])


# 测试函数，绘制整个圆弧并显示第 step 个点
def plot_3d_arc_with_step(step, start_point, end_point):
    total_steps = 100  # 总步数

    # 生成完整圆弧的所有点
    arc_points = np.array([get_arc_point(s, total_steps, start_point, end_point) for s in range(total_steps + 1)])

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制整个圆弧
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], label='3D Arc')

    # 标记圆弧上的特定点
    arc_step_point = get_arc_point(step, total_steps, start_point, end_point)
    ax.scatter(*arc_step_point, color='purple', label=f'Step {step} Point')

    # 标记起点、终点和圆心
    ax.scatter(*start_point, color='red', label='Start Point')
    ax.scatter(*end_point, color='blue', label='End Point')
    center = (start_point[:2] + end_point[:2]) / 2
    ax.scatter(center[0], center[1], (start_point[2] + end_point[2]) / 2, color='green', label='Center')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# 示例起点和终点
start_point = np.array([0, 0, 5])  # 起点
end_point = np.array([2, 6, 0])    # 终点

# 绘制并显示第 step 个点（例如 step = 30）
plot_3d_arc_with_step(30, start_point, end_point)


def get_target_point(step, start_point, end_point, total_steps=12000):
    # 计算圆心和半径
    center = (start_point[:2] + end_point[:2]) / 2
    radius = np.linalg.norm(start_point[:2] - end_point[:2]) / 2
    # 计算角度范围，从 0 到 π
    theta = np.pi * (step / total_steps)
    # 计算圆弧上的 x 和 y 坐标
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    # 计算旋转角度
    angle = np.arctan2(start_point[1] - end_point[1], start_point[0] - end_point[0])
    # 旋转矩阵
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    # 旋转并平移圆弧上的点
    arc_point_2d = np.dot(rotation_matrix, np.array([x, y]))
    arc_point_2d[0] += center[0]
    arc_point_2d[1] += center[1]
    z = start_point[2] + (end_point[2] - start_point[2]) * (step / total_steps)
    return np.array([arc_point_2d[0], arc_point_2d[1], z])
