@echo off

rem  固定学习率 修改控制频率 从 0.0003到0.0001都没有收敛

start cmd /k "activate py310 && cd /d E:\PyProjects\MARL-code-pytorch\4.MADDPG_MATD3_MPE && python MATD3_main_C3V1_Attention.py --algorithm MATD3 --lr_a 0.00015 --lr_c 0.00015 --batch_size 1024 --max_train_steps 10000000 --noise_decay_steps 8000000 --mark 9200 --N_drones 3"

start cmd /k "activate py310 && cd /d E:\PyProjects\MARL-code-pytorch\4.MADDPG_MATD3_MPE && python MATD3_main_C3V1_Attention.py --algorithm MATD3 --lr_a 0.0001 --lr_c 0.0001 --batch_size 1024 --max_train_steps 10000000 --noise_decay_steps 8000000 --mark 9201 --N_drones 3"

start cmd /k "activate py310 && cd /d E:\PyProjects\MARL-code-pytorch\4.MADDPG_MATD3_MPE && python MATD3_main_C3V1_Attention.py --algorithm MATD3 --lr_a 0.00008 --lr_c 0.00008 --batch_size 1024 --max_train_steps 10000000 --noise_decay_steps 8000000 --mark 9202 --N_drones 3"

start cmd /k "activate py310 && cd /d E:\PyProjects\MARL-code-pytorch\4.MADDPG_MATD3_MPE && python MATD3_main_C3V1_Attention.py --algorithm MATD3 --lr_a 0.00005 --lr_c 0.00005 --batch_size 1024 --max_train_steps 10000000 --noise_decay_steps 8000000 --mark 9203 --N_drones 3"

start cmd /k "activate py310 && cd /d E:\PyProjects\MARL-code-pytorch\4.MADDPG_MATD3_MPE && python MATD3_main_C3V1_Attention.py --algorithm MATD3 --lr_a 0.00003 --lr_c 0.00003 --batch_size 1024 --max_train_steps 10000000 --noise_decay_steps 8000000 --mark 9204 --N_drones 3"

rem 等待用户关闭所有窗口
pause
