import numpy as np


def potential_energy(obs_dict, num_agents, eta_att=1, eta_rep_agent=0.1, d0=1.0):
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
    delta_lm = obs_dict['target_pos'][:3]       # [3] 提取目标的相对位置
    dist_lm = obs_dict['target_pos'][3]         # 提取目标的距离
    if dist_lm > 0:
        unit_lm = delta_lm / dist_lm            # 引力单位方向
        F_att_abs = eta_att / (dist_lm ** 2)    # 根据需求调整
        F_att = unit_lm * F_att_abs
    else:
        F_att = np.zeros(3)

    # 计算斥力F_rep_agent
    F_rep_agent = np.zeros(3)
    other_pos = obs_dict['other_pos'].reshape((num_agents - 1, 4))
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


# 示例使用
if __name__ == "__main__":
    num_agents = 3
    obs_dict = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'rpy': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'ang_vel': np.array([0.0, 0.0, 0.0]),
        'target_pos': np.array([1.0, 1.0, 0.5, 2.0]),
        'other_pos': np.array([0., 0., 0.5, 0.5, 0., 0., 0.5, 0]).flatten(),
        'last_action': np.array([0.0, 0.0, 0.0, 0.0])
    }

    F = potential_energy(obs_dict, num_agents)
    print("Potential Energy F:", F)
