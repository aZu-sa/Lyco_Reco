import librosa
import librosa.display
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # No pictures displayed


def get_chroma_stft(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    return chroma_stft_mean, chroma_stft_var


def get_rms(y):
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    return rms_mean, rms_var


def get_spectral_centroid(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    return spectral_centroid_mean, spectral_centroid_var


def get_spectral_bandwidth(y, sr):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    return spectral_bandwidth_mean, spectral_bandwidth_var


def get_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rf_mean = np.mean(rolloff)
    rf_var = np.var(rolloff)
    return rf_mean, rf_var


def get_zero_crossing_rate(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr), np.var(zcr)


def get_harmony_and_perceptual(y):
    y_harm, y_perc = librosa.effects.hpss(y)
    return np.mean(y_harm), np.var(y_harm), np.mean(y_perc), np.var(y_perc)


def get_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


def get_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = list(map(np.mean, mfccs))
    mfcc_vars = list(map(np.var, mfccs))
    return mfcc_means, mfcc_vars


def get_eigenvector(path):
    """
        获取音频文件的特征向量
        path: 音频文件路径
        @return: 返回音频文件对应的特征向量
    """
    y, sr = librosa.load(path)
    y, _ = librosa.effects.trim(y)
    eigenvector = [*y.shape, *get_chroma_stft(y, sr), *get_rms(y), *get_spectral_centroid(y, sr),
                   *get_spectral_bandwidth(y, sr),
                   *get_rolloff(y, sr), *get_zero_crossing_rate(y), *get_harmony_and_perceptual(y), get_tempo(y, sr)]
    mfcc = get_mfcc(y, sr)
    for i in range(20):
        eigenvector.append(mfcc[0][i])
        eigenvector.append(mfcc[1][i])
    col = ["length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var", "spectral_centroid_mean",
           "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", "rolloff_var",
           "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var",
           "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", "mfcc3_mean",
           "mfcc3_var", "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", "mfcc6_var", "mfcc7_mean",
           "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", "mfcc10_mean", "mfcc10_var",
           "mfcc11_mean",
           "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", "mfcc13_var", "mfcc14_mean", "mfcc14_var",
           "mfcc15_mean",
           "mfcc15_var", "mfcc16_mean", "mfcc16_var", "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var",
           "mfcc19_mean",
           "mfcc19_var", "mfcc20_mean", "mfcc20_var"]
    return pd.DataFrame({col[idx]: eigenvector[idx] for idx in range(len(eigenvector))}, index=['1'])


if __name__ == "__main__":
    path = r"./resources/1.wav"
    eigenvector = get_eigenvector(path)
    for e in eigenvector:
        print(e)
