import numpy as np


def preprocess_rgb(image):
    """
    处理图像，将RGBA图像转换为目标检测掩码图像。

    参数:
    image (ndarray): 形状为 [N, 96, 128, 4] 的 RGBA 图像数据。

    返回:
    ndarray: 形状为 [N, 96, 128] 的目标检测掩码图像。
    """
    # 将 RGBA 图像的最后一维分离
    img_rgb = image[:, :, :, :3]

    # 定义颜色范围
    yellow_min = np.array([180, 180, 0], dtype=np.uint8)
    yellow_max = np.array([255, 255, 70], dtype=np.uint8)
    red_min = np.array([180, 0, 0], dtype=np.uint8)
    red_max = np.array([255, 70, 70], dtype=np.uint8)

    # 转换为浮点数以避免数据类型限制
    img_rgb = img_rgb.astype(np.float32)

    # 计算黄色和红色掩码
    yellow_mask = np.all((img_rgb >= yellow_min) & (img_rgb <= yellow_max), axis=-1)
    red_mask = np.all((img_rgb >= red_min) & (img_rgb <= red_max), axis=-1)

    # 创建掩码图像，黄色为1，红色为-1，其他为0
    mask_image = np.where(yellow_mask, 1., np.where(red_mask, -1., 0))

    return mask_image


# 测试
# 生成一个示例 RGBA 图像数据，形状为 [N, 96, 128, 4]
example_image = np.random.randint(0, 256, (3, 96, 128, 4), dtype=np.uint8)

# 调用处理函数
processed_image = preprocess_rgb(example_image)
print(processed_image.shape)  # 输出应为 [3, 96, 128]
