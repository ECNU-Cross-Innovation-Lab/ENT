import os
import re
import sys
import argparse
import json
import torch
import torchaudio
import numpy as np
import pickle as pkl

import utils

emotion_id = {'happy': 0, 'angry': 1, 'sad': 2}

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="/data/private/Dataset/ZED", help='path of IEMOCAP dataset')
parser.add_argument('--prodir', type=str, default=".", help='directory of IEMOCAP process')


def getOverlap(a, b):
    """ get the overlap length of two intervals

    Arguments
    ---------
    a : list
    b : list
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def get_labels(data, win_len=0.02, stride=0.02):
    """ make labels for training/test

    Arguments
    ---------
    data (dict): a dictionary that contains:
        {
            'wav': path,
            'duration': dur,
            'emotion': [{'emo': emo, 'start': s, 'end': e}],
            'transcription': trans,
        }
    win_len (float): the frame length used for frame-wise prediction
    stride (float): the frame length used for frame-wise prediction

    """
    emo_list = data["emotion"]
    assert len(emo_list) == 1

    duration = data["duration"]
    emotion = data["emotion"][0]["emo"]
    emo_start = data["emotion"][0]["start"]
    emo_end = data["emotion"][0]["end"]

    number_frames = int(duration / stride) + 1

    intervals = []
    labels = []
    if emo_start != 0:
        intervals.append([0.0, emo_start])
        labels.append("n")
    intervals.append([emo_start, emo_end])
    labels.append(emotion[0])
    if emo_end != duration:
        intervals.append([emo_end, duration])
        labels.append("n")

    start = 0.0
    frame_labels = []
    for i in range(number_frames):
        win_start = start + i * stride
        win_end = win_start + win_len

        # make sure that every sample exists in a window
        if win_end >= duration:
            win_end = duration
            win_start = max(duration - win_len, 0)

        for j in range(len(intervals)):
            if getOverlap([win_start, win_end], intervals[j]) >= 0.5 * (win_end - win_start):
                emo_frame = labels[j]
                break
        frame_labels.append(emo_frame)
        if win_end >= duration:
            break
    return intervals, labels, frame_labels


def generate_sed(indir, sed_pikdir):
    DataMap = []
    zed_json_path = os.path.join(indir, "ZED.json")
    with open(zed_json_path, "r") as f:
        all_dict = json.load(f)
    for key, value in all_dict.items():
        utterance_map = {}
        utterance_map['id'] = key
        wav_file_name = value['wav'].replace("datafolder", indir)

        # Process audio and label
        audio, audio_length = utils.load_wav(wav_file_name)
        utterance_map['audio'] = audio
        utterance_map['audio_length'] = audio_length
        utterance_map['duration'] = value['duration']

        # Process text
        text_line = value['transcription']
        text_line = text_line.upper()
        punc = "\[\]\.\,\?\!\-\:\;\""
        text_line = re.sub(r"[{0}]+".format(punc), " ", text_line)
        utterance_map['text'] = " ".join(text_line.split())

        # Process emotion
        utterance_map['interval'] = value['emotion'][0]
        utterance_map['emotion'] = emotion_id[value['emotion'][0]['emo']]
        try:
            intervals, labels, frame_labels = get_labels(value)
        except ValueError:
            print(f"Impossible to get labels for id {key} because the window is too large.")
            continue
        utterance_map['frame_label'] = frame_labels
        DataMap.append(utterance_map)
        

    with open(sed_pikdir, 'wb') as f:
        pkl.dump(DataMap, f)


if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.path
    nowdir = args.prodir
    sed_pikdir = f'{nowdir}/zed.pkl'
    # print(f'output pickle: {raw_pikdir}')
    generate_sed(indir, sed_pikdir)
