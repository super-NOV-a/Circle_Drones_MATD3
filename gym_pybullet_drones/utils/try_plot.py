import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 用于测试轨迹实时绘制
# 初始化参数
initial_position = np.array([-1.0, 0.0, 0.0])
final_position = np.array([0.0, 1.0, 0.0])
position = initial_position.copy()
total_time = 10  # 总时间10秒
time_step = 1.0  # 每秒更新一次位置
num_steps = int(total_time / time_step)

# 创建图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
point, = ax.plot([position[0]], [position[1]], [position[2]], 'bo')

# 设置图像范围
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

def update_point(new_position):
    global position
    # 更新位置
    position = new_position
    point.set_data(position[0], position[1])
    point.set_3d_properties(position[2])
    plt.draw()


def get_new_position(self, step):
    # 计算半圆轨迹上的点
    initial_position = self.INIT_Target
    final_position = self.END_Target
    theta = np.pi * step / 1000  # 从0到π
    radius = np.linalg.norm(final_position[:2] - initial_position[:2]) / 2  # 半径
    center = (initial_position[:2] + final_position[:2]) / 2  # 圆心
    x = center[0] + radius * np.cos(theta)  # x坐标
    y = center[1] + radius * np.sin(theta)  # y坐标
    z = initial_position[2] + (final_position[2] - initial_position[2]) * step / 1000  # z坐标线性插值
    return np.array([x, y, z])

def animate(i):
    new_position = get_new_position(i, initial_position, final_position)
    update_point(new_position)


# 创建动画
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=time_step * 1000, repeat=False)
plt.show()