import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_and_plot_results(save_file_path):
    # 打开文件读取内容
    with open(save_file_path, 'r') as f:
        lines = f.readlines()

    # 解析目标位置和无人机轨迹
    target_pos = []
    agent_states = []
    reading_target = False
    agent_id = -1  # 初始化为无效值，后续会更新为无人机 ID

    for line in lines:
        # 忽略空行
        if not line.strip():
            continue

        # 检查是否是目标轨迹
        if line.startswith("# Target Trajectories"):
            reading_target = True
            continue
        elif line.startswith("# Agent Trajectories"):
            reading_target = False
            continue
        elif line.startswith("Agent"):
            agent_id += 1  # 遇到一个新无人机时增加 ID
            agent_states.append([])  # 为该无人机创建一个轨迹列表
            continue

        # 分割并解析坐标
        coords = line.strip().split(", ")
        coords = [float(c) for c in coords]

        if reading_target:
            target_pos.append(coords)
        else:
            agent_states[agent_id].append(coords)  # 添加到当前无人机的轨迹中

    # 将列表转换为 numpy 数组
    target_pos = np.array(target_pos)
    agent_states = [np.array(agent) for agent in agent_states]

    # 创建图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 定义不同的颜色，用于区分无人机和目标
    colors = ['r', 'g', 'b', 'y']  # 三个无人机和一个目标位置

    # 绘制目标点轨迹（目标位置曲线）
    ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
            color=colors[3], label='Target Position', linestyle='--')

    # 绘制每个无人机的轨迹
    for agent_id, agent_traj in enumerate(agent_states):
        ax.plot(agent_traj[:, 0], agent_traj[:, 1], agent_traj[:, 2],
                color=colors[agent_id % len(colors)], label=f'Agent {agent_id}')

        # 突出显示每个无人机轨迹的起始点
        ax.scatter(agent_traj[0, 0], agent_traj[0, 1], agent_traj[0, 2],
                   color=colors[agent_id], s=100, marker='o')  # 起始点使用大号标记符号

    # 添加图例
    ax.legend()
    # 显示图像
    plt.show()


# 调用函数并传入保存文件路径
save_file_path = "c3v1A_9200/0.txt"  # 替换为实际的文件路径
load_and_plot_results(save_file_path)
