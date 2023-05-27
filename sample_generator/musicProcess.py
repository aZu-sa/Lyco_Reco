import matplotlib

import pylab
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import os, re

matplotlib.use('Agg')  # No pictures displayed


def devide(genre):
    """
    从resources文件夹读取wav文件,
    将每个音频文件以30秒为间隔切割,
    保存到audio_clips/#{genre}目录下
    """
    # # 循环目录下所有文件
    for each in os.listdir(r"./resources/"):  # 循环目录

        filename = re.findall(r"(.*?)\.wav", each)  # 取出.wav后缀的文件名
        print(each)
        if each:

            wav = AudioSegment.from_file('./resources/{}'.format(each), "wav")  # 打开wav文件
            # mp3[17*1000+500:].export(filename[0], format="mp3") # 切割前17.5秒并覆盖保存，与以下代码不可同时使用
            size = 30000  # 切割的毫秒数 10s=10000

            chunks = make_chunks(wav, size)  # 将文件切割为10s一块

            for i, chunk in enumerate(chunks):
                chunk_name = "{}-{}.wav".format(each.split(".")[0], i)  # 也可以自定义名字
                print(chunk_name)
                chunk.export('./audio_clips/{}/{}'.format(genre, chunk_name), format="wav")  # 新建的保存文件夹


def generateMelSpectrogram(genre):
    """
    从audio_clips文件夹读取wav文件 (mp3格式在windows下不支持)，
    生成每个音频文件梅尔变换后的频谱图，
    保存到melSpectrogram/#{genre}目录下。
    """

    read_path = r"./audio_clips/"
    output_path = r"./melSpectrogram/{}/".format(genre)
    files = os.listdir(read_path)

    for file_name in files:
        print(file_name)
        sig, fs = librosa.load(read_path + file_name)

        # make pictures name
        save_path = output_path + file_name.split('.')[0]

        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)

    pylab.close()


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


def get_harmony(y):
    harmony = librosa.effects.harmonic(y)
    return np.mean(harmony), np.var(harmony)


def get_perceptr():
    pass


def get_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


if __name__ == "__main__":
    pass
