import time
import numpy as np


# 假设我们有一个函数可以获取陀螺仪的实时数据
# 这个函数应该返回一个包含x, y, z轴角速度的元组
def get_gyroscope_data():
    # 这里是模拟数据，实际应用中需要替换为真实的传感器读取
    return (0.001, 0.002, -0.001)  # 示例数据：微小的角速度


# 设置一个阈值，用于判断陀螺仪数据是否表示静止
# 这个阈值需要根据实际情况进行调整
THRESHOLD = 0.01

# 设置一个时间窗口，用于计算平均角速度
# 如果在这个时间窗口内，平均角速度小于阈值，则认为设备处于静止状态
TIME_WINDOW = 10  # 单位：秒
SAMPLE_RATE = 1  # 每秒采样次数

# 初始化变量
gyro_data_window = []
start_time = time.time()

# 模拟持续获取陀螺仪数据并判断设备状态
while True:
    # 获取当前陀螺仪数据
    gx, gy, gz = get_gyroscope_data()

    # 计算当前角速度的模（矢量长度）
    current_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # 将当前数据添加到时间窗口内
    gyro_data_window.append(current_magnitude)

    # 如果时间窗口已满，则计算平均角速度并判断设备状态
    if len(gyro_data_window) >= TIME_WINDOW * SAMPLE_RATE:
        # 计算平均角速度
        average_magnitude = sum(gyro_data_window) / len(gyro_data_window)

        # 判断设备是否处于静止状态
        if average_magnitude < THRESHOLD:
            print("设备处于静止状态，可能用户正在睡眠。")
        else:
            print("设备正在移动。")

        # 清空时间窗口以便下一次计算
        gyro_data_window = []
        start_time = time.time()

    # 等待下一次采样
    time.sleep(1 / SAMPLE_RATE)