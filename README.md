# Circle_Drones_MATD3
用于保留我的代码，自定义circle_spread环境下，使用MATD3训练

* ~~第一次训练前需要保证 model文件夹下存在circle文件夹（todo 自动生成）~~

~~速度映射还有些问题，可以考虑修改为ctrl中更新target_pos和rpy，而不仅仅是target_vel~~

9/19：本次更新时使用位姿信息，环境中新增势能Fs计算，网络复现并稍微修改了师姐的势能+注意力网络（POMMAC），但是运行速度较低。

10/5：观测去掉了势能，使用注意力、图两部分代码代替师姐的网络，运行速度快，使用分散的critic代替原本的中心化critic

使用时运行：

```
4.MADDPG_MATD3_MPE/MATD3_main_CircleSpread.py	(MATD3+基准模型)
4.MADDPG_MATD3_MPE/MATD3_main_CircleSpread_Attention.py	(MATD3+注意力模型)
4.MADDPG_MATD3_MPE/MATD3_main_CircleSpread_Graph.py	(MATD3+图模型)
```

测试时使用：

```
4.MADDPG_MATD3_MPE/try_MATD3_circle.py  (测试指定模型)
4.MADDPG_MATD3_MPE/read_load_paths.py   (绘制指定路径)
```

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

