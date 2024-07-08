import numpy as np


class TargetUpdater:
    def __init__(self, num_drones):
        self.NUM_DRONES = num_drones
        self.target_pos = np.zeros((self.NUM_DRONES, 3))
        self.target_pos[:, 2] = 1  # 初始目标位置在 (0, 0, 1)

    def update_target_pos(self, step_counter):
        # 渐变程度与步数计数器相关
        gradient = np.arctan(step_counter)/ 100   #

        # 随机选择渐变方向
        direction = np.random.uniform(-1, 1, size=(3,))
        direction /= np.linalg.norm(direction)  # 归一化

        # 根据渐变程度和方向更新目标位置
        self.target_pos += gradient * direction

        # 投影到所需的范围内
        self.target_pos[:, 0:2] = np.clip(self.target_pos[:, 0], -1, 1)
        # self.target_pos[:, 1] = np.clip(self.target_pos[:, 1], -1, 1)
        self.target_pos[:, 2] = np.clip(self.target_pos[:, 2], 0, 1)

        return self.target_pos

# 示例循环输出
target_updater = TargetUpdater(num_drones=1)  # 初始化目标位置更新器
for step_counter in range(100000):  # 假设循环10次
    target_pos = target_updater.update_target_pos(step_counter)
    print("Step:", step_counter, "Target Position:", target_pos)
