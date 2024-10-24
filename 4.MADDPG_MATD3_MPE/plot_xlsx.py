import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pylab import mpl

# 设置字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

warnings.filterwarnings('ignore')

# 平滑函数
def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm) * 1.0
            d = np.convolve(y, d, "same") / np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

file_path = 'C:/Users/Niu/Desktop/经验回放对比2.xlsx'
data = pd.read_excel(file_path)

# 提取数据
method1_rewards = data.iloc[1:, 0].astype(float).values
method2_rewards = data.iloc[1:, 1].astype(float).values

# 将数据存储到列表中
rewards_data = [method1_rewards, method2_rewards]
y_data = smooth(rewards_data, 3)

# 创建x轴数据
x_data = np.arange(len(method1_rewards))

# # 设置绘图风格
# sns.set(style="darkgrid", font_scale=1.5)
#
# # 绘制对比图
# plt.figure(figsize=(12, 6))
# sns.lineplot(x=x_data, y=y_data[0], color='blue', label='多步经验回放', linestyle='-')
# sns.lineplot(x=x_data, y=y_data[1], color='orange', label='基准经验回放', linestyle='-')
#
# # 添加图例、标题和坐标轴标签
# plt.title('不同经验回放机制收敛性对比', fontsize=16)
# plt.xlabel('Episodes', fontsize=14)
# plt.ylabel('Rewards', fontsize=14)
# plt.legend()
# plt.ylim(bottom=-2000)  # 设置Y轴下限为0
# plt.tight_layout()

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_data, y=y_data[0], mode='lines', name='多步经验回放', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x_data, y=y_data[1], mode='lines', name='基准经验回放', line=dict(color='orange')))

fig.update_layout(title='不同经验回放机制收敛性对比', xaxis_title='Episodes', yaxis_title='Rewards')
fig.show()


# plt.savefig('convergence_comparison_seaborn.png', dpi=720)
plt.show()

