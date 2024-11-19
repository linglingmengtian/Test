import warnings

import librosa
import numpy as np
import os
import soundfile as sf
import ifvioice
from datetime import timedelta


def extract_features(file_name):
    try:
        # 加载音频
        audio, sample_rate = librosa.load(file_name, sr=None)
        audio = librosa.util.normalize(audio)  # 音量归一化

        # 提取MFCC特征
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
        except librosa.Error as e:
            print(f"Error extracting MFCCs from {file_name}: {e}")
            return None

        # 提取色谱图特征
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)

        # 提取MFCC的一阶和二阶导数
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        delta_mfccs_mean = np.mean(delta_mfccs.T, axis=0)
        delta2_mfccs_mean = np.mean(delta2_mfccs.T, axis=0)

        try:
            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")

                # Trigger a warning.
                pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)

                # Verify some things
                checked = []
                for ww in w:
                    checked.append(str(ww.message))
                    if "Trying to estimate tuning from empty frequency set" in str(ww.message):
                        print(f"Warning: {ww.message} for file {file_name}")
                        pitch = None
                        break
                else:
                    pitch = []
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch.append(pitches[index, t])

            pitch_mean = np.mean(pitch) if pitch is not None and pitch else None
        except librosa.Error as e:
            print(f"Error extracting pitch from {file_name}: {e}")
            pitch_mean = None

            # 组合特征向量
        feature_vector = np.concatenate(
            (mfccs_mean, chroma_mean, delta_mfccs_mean, delta2_mfccs_mean,
             [pitch_mean] if pitch_mean is not None else [0]))

        return feature_vector
    except Exception as e:
        print(f"An unexpected error occurred while processing file: {file_name}\nError: {e}")
        return None

# 加载模型和scaler
scaler = ifvioice.get_scaler()
model = ifvioice.get_model()

# 设置录音文件路径和分割时间间隔
recording_path = 'SPK001-01.wav'
segment_duration = 10  # 以秒为单位

# 加载录音文件
y, sr = librosa.load(recording_path, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# 初始化打鼾统计
snore_count = 0
snore_duration = timedelta(0)
snore_intensity_sum = 0

# 临时文件路径
temp_file_path = 'temp_segment.wav'

# 遍历每个片段
for start_time in range(0, int(duration), segment_duration):
    end_time = min(start_time + segment_duration, int(duration))

    # 提取片段的音频数据
    segment = y[start_time * sr:end_time * sr]

    # 保存片段为临时文件
    sf.write(temp_file_path, segment, sr)

    # 提取特征
    features = extract_features(temp_file_path)
    if features is not None:
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
        prediction = model.predict(features_scaled)

        # 如果预测为打鼾
        if prediction == 1:
            snore_count += 1
            snore_duration += timedelta(seconds=segment_duration)

            # 量化打鼾强度，这里使用特征的L2范数作为强度的简单度量
            snore_intensity = np.linalg.norm(features)
            snore_intensity_sum += snore_intensity

# 计算平均打鼾强度
if snore_count > 0:
    average_snore_intensity = snore_intensity_sum / snore_count
else:
    average_snore_intensity = 0

# 打印结果
print(f"打鼾次数: {snore_count}")
print(f"打鼾持续时间: {snore_duration}")
print(f"平均打鼾强度: {average_snore_intensity:.2f}")

# 清理临时文件
os.remove(temp_file_path)