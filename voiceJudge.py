import pyaudio
import numpy as np
import time

# 设置参数
FORMAT = pyaudio.paInt16  # 数据格式
CHANNELS = 1  # 单声道
RATE = 44100  # 采样率
CHUNK = 1024  # 每次读取的音频流长度
THRESHOLD = 500  # 声音阈值，低于此值认为环境安静
WINDOW_SIZE = 5  # 时间窗口大小，单位：秒
CHUNKS_PER_WINDOW = RATE * WINDOW_SIZE // CHUNK  # 每个窗口包含的块数

# 初始化PyAudio对象
audio = pyaudio.PyAudio()

# 打开麦克风流
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("开始监听...")

quiet_count = 0  # 记录连续安静的时间窗口数
try:
    while True:
        for _ in range(CHUNKS_PER_WINDOW):
            # 读取音频数据
            data = stream.read(CHUNK)
            # 将音频数据转换为numpy数组
            audio_data = np.frombuffer(data, dtype=np.int16)
            # 计算音频信号的均方根（RMS）值
            rms = np.sqrt(np.mean(audio_data ** 2))

            # 如果发现声音超过阈值，重置安静计数器
            if rms >= THRESHOLD:
                quiet_count = 0
                break
        else:
            # 如果整个窗口内的RMS都低于阈值，增加安静计数器
            quiet_count += 1

        # 根据安静计数器判断睡眠状态
        if quiet_count >= 10:  # 可以根据需要调整判断阈值
            print("环境持续安静，可能已经入睡。")
        else:
            print("环境有声音，未检测到睡眠状态。")

        # 可以在这里添加一个适当的sleep来减少CPU使用率
        time.sleep(0.1)

except KeyboardInterrupt:
    print("停止监听。")
finally:
    # 关闭流和PyAudio对象
    stream.stop_stream()
    stream.close()
    audio.terminate()