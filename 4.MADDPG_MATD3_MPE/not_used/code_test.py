import os

# 设置文件夹路径
folder_path = "E:\PyProjects\MARL-code-pytorch\\4.MADDPG_MATD3_MPE\\agent_paths/c3v1A_9200+1_0.13"

# 初始化计数器
win_true_count = 0  # 统计 "胜" 位置 True 的次数
win_false_count = 0  # 统计 "胜" 位置 False 的次数
achieve_true_count = 0  # 统计 "成" 位置 True 的次数
achieve_false_count = 0  # 统计 "成" 位置 False 的次数

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # 假设文件名格式是 X_胜Y_成Z.txt
        parts = filename.split('_')

        # 统计 "胜" 位置 (Y) 和 "成" 位置 (Z) 的 True 和 False
        win_status = parts[1][1:]  # "胜" 后面的 True 或 False
        achieve_status = parts[2][1:].replace('.txt', '')  # "成" 后面的 True 或 False

        if win_status == "True":
            win_true_count += 1
        elif win_status == "False":
            win_false_count += 1

        if achieve_status == "True":
            achieve_true_count += 1
        elif achieve_status == "False":
            achieve_false_count += 1

print(f"胜位置 True 的次数: {win_true_count}")
print(f"胜位置 False 的次数: {win_false_count}")
print(f"成位置 True 的次数: {achieve_true_count}")
print(f"成位置 False 的次数: {achieve_false_count}")
