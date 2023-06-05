import os
import re

import joblib
from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sample_generator.musicProcess as mp

import warnings

warnings.filterwarnings("ignore")


def max_three(iterable, key):
    __iter = sorted(iterable, key=key, reverse=True)
    return __iter[-3:]


def sort_key(x):
    key = 0
    base = 1
    for idx in range(5, len(x)):
        if '0' <= x[-idx] <= '9':
            key = int(x[-idx]) * base
            base *= 10
    return key


def generate_weight(data_len):
    weight = [[n] for n in range(1, data_len + 1)]
    for idx in range(data_len // 2, data_len):
        weight[idx] = weight[data_len - idx - 1]
    weight.insert(0, [0])
    weight.append([0])
    return weight


def divide():
    read_path = r"./sample_generator/resources/"
    output_path = r"./sample_generator/audio_clips/"
    for each in os.listdir(read_path):
        if each:
            wav = AudioSegment.from_file('{}{}'.format(read_path, each), "wav")
            size = 30000
            chunks = make_chunks(wav, size)
            for i, chunk in enumerate(chunks):
                chunk_name = "{}-{}.wav".format(each.split(".")[0], i)
                chunk.export('{}{}'.format(output_path, chunk_name), format="wav")


def predict():
    read_path = r"./sample_generator/audio_clips/"
    model_data_path = r"./saved-model/2023-6-5-12-05-ada.pkl"
    model = joblib.load(model_data_path)
    # print(model.classes_)
    proba = [0 for _ in range(len(model.classes_))]

    stdsc = StandardScaler()
    weight = generate_weight(len(os.listdir(read_path)))
    weight = stdsc.fit_transform(weight)
    m = min(weight, key=lambda x: x[0])[0]
    for idx in range(len(weight)):
        weight[idx][0] -= m
    weight = weight.T[0]
    print(weight)
    _idx = 1
    clips = os.listdir(read_path)
    clips = sorted(clips, key=sort_key)
    for each in clips:
        # print(each)
        eigenvector = mp.get_eigenvector("{}{}".format(read_path, each))
        # print(eigenvector)
        pre = model.predict_proba(eigenvector)[0]
        # print(pre)
        for idx in range(len(pre)):
            proba[idx] += pre[idx] * weight[_idx]
        _idx += 1

    return {model.classes_[idx]: proba[idx] for idx in range(len(model.classes_))}


divide()
# print()
proba = predict()
# print(proba)
print(max_three(proba.items(), key=lambda x: x[1]))
