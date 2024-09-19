import numpy as np
import matplotlib.pyplot as plt


def read_and_plot_results(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    target_points = []
    agent_trajectories = []
    current_agent_traj = []
    num_agents = 0
    parsing_agent = False

    for line in lines:
        line = line.strip()

        # 跳过空行或无效行
        if not line or line.startswith("#"):
            continue

        # 读取目标点
        if ":" in line and "Target" not in line:
            if "trajectory" not in line:
                target_data = line.split(":")[1].strip()
                target_points.append([float(x) for x in target_data.split(",")])
            else:
                if parsing_agent:  # 如果之前在解析某个智能体，保存之前的数据
                    agent_trajectories.append(np.array(current_agent_traj))
                    current_agent_traj = []
                num_agents += 1  # 新智能体的轨迹开始
                parsing_agent = True
        else:
            # 读取智能体轨迹点
            traj_point = [float(x) for x in line.split(",") if x]
            if len(traj_point) == 3:  # 确保是(x, y, z)坐标
                current_agent_traj.append(traj_point)

    # 确保最后一个智能体的轨迹被保存
    if current_agent_traj:
        agent_trajectories.append(np.array(current_agent_traj))

    # 创建绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 定义不同颜色
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # 绘制目标点，并使用不同的颜色
    for idx, target in enumerate(target_points):
        ax.scatter(target[0], target[1], target[2],
                   color=colors[idx % len(colors)],  # 使用不同的颜色
                   s=100, marker='x', label=f'Target {idx}')  # 使用不同的标记符号

    # 绘制轨迹并标记起点
    for agent_id, trajectory in enumerate(agent_trajectories):
        trajectory = np.array(trajectory)
        if trajectory.size > 0:
            # 绘制轨迹线
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color=colors[agent_id % len(colors)], label=f'Agent {agent_id} trajectory')

            # 标记起点（使用更大的点和不同的标记符号）
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                       color=colors[agent_id % len(colors)], s=100, marker='o', label=f'Agent {agent_id} start')

    # 设置标签和标题
    ax.set_title('Agent Positions and Actions Over Time')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.tight_layout()
    plt.show()


# 使用示例
read_and_plot_results('agent_paths/good_example.txt')
