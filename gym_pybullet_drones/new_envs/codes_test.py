import numpy as np
import matplotlib.pyplot as plt


def detect_yellow_region(image, min_area=20):
    """
    检测图像中是否存在一定区域的黄色
    :param image: 输入图像，形状为 (128, 96, 4)
    :param min_area: 最小黄色区域面积
    :return: 是否存在一定区域的黄色
    """
    yellow_min = np.array([180, 180, 0], dtype=np.uint8)
    yellow_max = np.array([255, 255, 70], dtype=np.uint8)

    # 提取RGB通道
    img_rgb = image[:, :, :3]

    # 识别黄色区域
    yellow_mask = np.all((img_rgb >= yellow_min) & (img_rgb <= yellow_max), axis=-1)

    # 计算黄色区域的面积
    yellow_area = np.sum(yellow_mask)

    # 判断是否存在足够大的黄色区域
    return yellow_mask, yellow_area >= min_area


def create_yellow_detection_image(image):
    """
    创建一个标记黄色区域的图像
    :param image: 输入图像，形状为 (128, 96, 4)
    :return: 标记黄色区域的图像，形状为 (128, 96, 3)
    """
    yellow_mask, _ = detect_yellow_region(image)

    # 创建一个全白色的图像
    Y = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

    # 将黄色区域标记为黄色
    Y[yellow_mask] = [255, 255, 0]

    return Y


# 示例图像
np.random.seed(42)  # 固定随机种子以便重现
image = np.random.randint(0, 256, (128, 96, 4), dtype=np.uint8)  # 示例RGBA图像

# 创建标记黄色区域的图像
Y = create_yellow_detection_image(image)

# 显示原始图像和标记黄色区域的图像
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image[:, :, :3])
axs[0].set_title("Original Image")
axs[0].axis('off')  # 隐藏坐标轴

axs[1].imshow(Y)
axs[1].set_title("Yellow Detection Image")
axs[1].axis('off')  # 隐藏坐标轴

plt.show()
