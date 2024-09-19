import torch
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CircleSpread import CircleSpreadAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

Env_name = 'circle'  # 'spread3d', 'simple_spread'
action = 'mix'

env = CircleSpreadAviary(gui=True, num_drones=1, obs=ObservationType('kin_target'),
                         act=ActionType(action),
                         ctrl_freq=30,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                         need_target=True, obs_with_act=True)
done = False
print('4维混合动作：a[0] 处理前进后退（x轴方向） a[1] 处理上升下降（z轴方向）a[2] 处理航向偏转（yaw角） a[3] 处理是否保持静止')
while not done:
    # 输入混合动作
    action_input = input('输入混合动作: 0-3, 空格隔开: ')
    try:
        # 解析用户输入为动作数组
        parts = action_input.split(' ')
        cont_action = np.array([float(x) for x in parts[:3]])
        discrete_action = int(parts[3])

        if len(cont_action) != 3:
            raise ValueError("连续动作必须包含3个值")
        if discrete_action not in [0, 1]:
            raise ValueError("离散动作必须为0或1")

        action = (cont_action, discrete_action)
    except ValueError as e:
        print(f"输入错误: {e}")
        continue

    # 将动作传递给环境
    for _ in range(10):
        rpm = env.try_mixed(action)
        clipped_action = np.reshape(rpm, (1, 4))
        env.apply_physics(clipped_action[0, :], 0)
        p.stepSimulation()

# 关闭环境
env.close()