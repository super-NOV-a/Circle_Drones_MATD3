# Circle_Drones_MATD3
用于保留我的代码，自定义circle_spread环境下，使用MATD3训练

* 第一次训练前需要保证 model文件夹下存在circle文件夹（todo 自动生成）

~~离散动作训练起来依旧困难（状态空间到动作空间的映射太大了），目前简化的内容包括：图像识别—读取黄色红色像素，输入简化为像素输入；动作去掉上升下降了，无人机们全都保持某一高度。并且对应修改了网络。~~

~~速度映射还有些问题，可以考虑修改为ctrl中更新target_pos和rpy，而不仅仅是target_vel~~

9/19：本次更新时使用位姿信息，环境中新增势能Fs计算，网络复现并稍微修改了师姐的势能+注意力网络（POMMAC），但是运行速度较低。

此仓库是在下列仓库基础上创建的：[Lizhi-sjtu/MARL-code-pytorch: Concise pytorch implements of MARL algorithms, including MAPPO, MADDPG, MATD3, QMIX and VDN. (github.com)](https://github.com/Lizhi-sjtu/MARL-code-pytorch)

目前本人使用的**python3.10**环境的package主要包括：

```
Package                 Version
----------------------- ----------------
gym                     0.10.5
gymnasium               0.28.1
matplotlib              3.7.0
numpy                   1.24.0
pip                     23.3.1
pybullet                3.2.5
torch                   2.2.2+cu118
```

