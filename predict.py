import os
import re

import joblib
from pydub import AudioSegment
from pydub.utils import make_chunks

import sample_generator.musicProcess as mp

import warnings

warnings.filterwarnings("ignore")


def divide():
    read_path = r"./sample_generator/resources/"
    output_path = r"./sample_generator/audio_clips/"
    for each in os.listdir(read_path):  # 循环目录
        print(each)
        if each:

            wav = AudioSegment.from_file('{}{}'.format(read_path, each), "wav")  # 打开wav文件
            # mp3[17*1000+500:].export(filename[0], format="mp3") # 切割前17.5秒并覆盖保存，与以下代码不可同时使用
            size = 30000  # 切割的毫秒数 10s=10000

            chunks = make_chunks(wav, size)  # 将文件切割为30s一块
            for i in range(len(chunks)):
                if i == len(chunks) - 1:
                    break
                chunk_name = "{}-{}.wav".format(each.split(".")[0], i)  # 也可以自定义名字
                print(chunk_name)
                chunks[i].export('{}{}'.format(output_path, chunk_name), format="wav")  # 新建的保存文件夹
            # for i, chunk in enumerate(chunks):
            #     chunk_name = "{}-{}.wav".format(each.split(".")[0], i)  # 也可以自定义名字
            #     print(chunk_name)
            #     chunk.export('{}{}'.format(output_path, chunk_name), format="wav")  # 新建的保存文件夹


def predict():
    read_path = r"./sample_generator/audio_clips/"
    model_data_path = r"./saved-model/2023-6-5-10-54-knn.pkl"
    rf = joblib.load(model_data_path)
    print(rf.classes_)
    proba = [0 for _ in range(len(rf.classes_))]
    for each in os.listdir(read_path):
        print(each)
        eigenvector = mp.get_eigenvector("{}{}".format(read_path, each))
        # print(eigenvector)
        pre = rf.predict_proba(eigenvector)
        print(pre)
        for idx in range(len(pre[0])):
            proba[idx] += pre[0][idx]

    return {rf.classes_[idx]: proba[idx] for idx in range(len(rf.classes_))}

def cleanAudio_clips(): # 新增自动清理
    path = r"./sample_generator/audio_clips/"
    for each in os.listdir(path):  # 循环目录
        # filename = re.findall(r"(.*?)\.wav", each)  # 取出.wav后缀的文件名
        os.remove("{}{}".format(path, each))

divide()
print()
proba = predict()
print(proba)
print(max(proba.items(), key=lambda x: x[1]))
cleanAudio_clips()
