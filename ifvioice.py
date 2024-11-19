import os
import librosa
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings


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

            # STFT特征（10维）
            stft = np.abs(librosa.stft(audio))
            stft_mean = np.mean(stft)
            stft_var = np.var(stft)
            stft_max = np.max(stft)
            stft_min = np.min(stft)
            # 计算中心矩（第1-6阶）
            stft_central_moments = [moment(stft, order=i, axis=None, nan_policy='omit') for i in range(1, 7)]

            # Mel Spectrogram特征（5维）
            mel_spect = librosa.feature.melspectrogram(audio, sr=sample_rate)
            mel_spect_mean = np.mean(mel_spect)
            mel_spect_var = np.var(mel_spect)
            mel_spect_max = np.max(mel_spect)
            mel_spect_min = np.min(mel_spect)
            mel_spect_energy = np.sum(mel_spect)

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


# 数据集路径
origin_dir = 'origin'

features = []
labels = []

# 遍历文件夹，提取特征
for label, folder in enumerate([os.path.join(origin_dir, '0'), os.path.join(origin_dir, '1')]):
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if file_name.endswith('.wav'):
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(label)

# 转换为NumPy数组
features = np.array(features)
labels = np.array(labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练XGBoost模型
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with XGBoost and additional features: {accuracy:.2f}')


def get_scaler():
    return scaler


def get_model():
    return model