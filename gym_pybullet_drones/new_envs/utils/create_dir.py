import os

env_name = 'c3v1A'
model_dir = 'model'

def check_create_dir(env_name, model_dir):
    # 检查model文件夹是否存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 检查c3v1A文件夹是否存在
    folder_path = os.path.join(model_dir, env_name)
    if os.path.exists(folder_path):
        print('当前文件夹存在')
    else:
        os.makedirs(folder_path)
        print(f'创建文件夹: {folder_path}')


if __name__ == "__main__":
    check_create_dir(env_name, model_dir)
