import os
import random
import argparse
import numpy as np
import pickle as pkl
from pydub import AudioSegment
from transformers import Wav2Vec2Config, Wav2Vec2Model
import utils
from toke import read_symbol_table

config = Wav2Vec2Config()
model = Wav2Vec2Model(config)

symbol_table = {}
emotion_id = {'hap': 0, 'ang': 1, 'sad': 2, 'neu': 3, 'exc': 0}
combinations = ["neu_emo", "emo_neu", "neu_emo_neu", "emo_emo"]
probabilities = np.array([0.25, 0.25, 0.25, 0.25])
speaker_dict = {
    'IEMOCAP': ["Ses01F", "Ses01M", "Ses02F", "Ses02M", "Ses03F", "Ses03M", "Ses04F", "Ses04M", "Ses05F", "Ses05M"]
}

parser = argparse.ArgumentParser()
parser.add_argument('--prodir', type=str, default=".", help='directory of IEMOCAP process')
parser.add_argument('--dataset', type=str, help='dataset to be mixed')


def getLength(frame_label):
    frame_label_length = []
    length = 0
    now_label = frame_label[0]
    for label in frame_label:
        if label == now_label:
            length += 1
        else:
            frame_label_length.append(length)
            length = 0
            now_label = label
    frame_label_length.append(length)
    return frame_label_length


def tokenize(tokens):
    tlabel = []
    for ch in tokens:
        if ch in symbol_table:
            tlabel.append(symbol_table[ch])
        elif '<unk>' in symbol_table:
            tlabel.append(symbol_table['<unk>'])
    return tlabel


def getOverlap(a, b):
    """ get the overlap length of two intervals

    Arguments
    ---------
    a : list
    b : list
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def get_feat_extract_output_lengths(input_length):
    """
        Computes the output length of the convolutional layers
        """
    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        input_length = _conv_out_length(input_length, kernel_size, stride)
    return input_length


def get_labels(data, length, win_len=0.02, stride=0.02):
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
        labels.append(3)
    intervals.append([emo_start, emo_end])
    labels.append(emotion)
    if emo_end != duration:
        intervals.append([emo_end, duration])
        labels.append(3)

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
    while len(frame_labels) < length:
        frame_labels.append(emo_frame)
    if len(frame_labels) > length:
        frame_labels = frame_labels[:length]
    return labels, frame_labels


def ConcatRaw(combdir, raw_pikdir, out_pikdir, dataset):
    with open(raw_pikdir, 'rb') as f:
        DataList = pkl.load(f)
    repos = speaker_dict[dataset]
    CombList = []
    for repo in repos:
        emotion_list = [
            utterance_map for utterance_map in DataList
            if repo in utterance_map['audio'] and f"_{repo[-1]}" in utterance_map['audio'] and utterance_map['label'] != 3
        ]
        neutral_list = [
            utterance_map for utterance_map in DataList
            if repo in utterance_map['audio'] and f"_{repo[-1]}" in utterance_map['audio'] and utterance_map['label'] == 3
        ]
        random.shuffle(emotion_list)
        random.shuffle(neutral_list)
        neutral_list = neutral_list * 20
        combine_path = os.path.join(combdir, "combined", repo)
        if not os.path.exists(combine_path):
            os.makedirs(combine_path)

        while len(emotion_list) > 0:
            combination = np.random.choice(combinations, p=probabilities.ravel())
            utterance_map = {}
            if combination == "neu_emo":
                neutral_sample = neutral_list[0]['audio']
                emo_sample = emotion_list[0]['audio']
                emotion = emotion_list[0]['label']

                neutral_input = AudioSegment.from_wav(neutral_sample).set_frame_rate(16000)
                emotion_input = AudioSegment.from_wav(emo_sample).set_frame_rate(16000)

                emotion_input += neutral_input.dBFS - emotion_input.dBFS
                combined_input = neutral_input + emotion_input

                out_name = os.path.join(combine_path, neutral_sample.split("/")[-1][:-4] + "_" + emo_sample.split("/")[-1])
                combined_input.export(out_name, format="wav")

                data_json = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [{
                        "emo": emotion,
                        "start": len(neutral_input) / 1000,
                        "end": len(combined_input) / 1000,
                    }],
                }

                audio, audio_length = utils.load_wav(out_name)
                utterance_map['audio'] = audio
                utterance_map['audio_length'] = get_feat_extract_output_lengths(audio_length)
                utterance_map['text'] = " ".join([neutral_list[0]['text'], emotion_list[0]['text']])
                utterance_map['tlabel'] = tokenize(utterance_map['text'])
                utterance_map['frame_tlabel'] = [3] * (len(neutral_list[0]['text']) + 2) + [emotion] * len(emotion_list[0]['text'])
                utterance_map['label'], utterance_map['frame_label'] = get_labels(data_json, utterance_map['audio_length'])
                utterance_map['frame_tlabel_length'] = getLength(utterance_map['frame_tlabel'])
                utterance_map['frame_label_length'] = getLength(utterance_map['frame_label'])

                neutral_list = neutral_list[1:]
                emotion_list = emotion_list[1:]

            elif combination == "emo_neu":
                neutral_sample = neutral_list[0]['audio']
                emo_sample = emotion_list[0]['audio']
                emotion = emotion_list[0]['label']

                neutral_input = AudioSegment.from_wav(neutral_sample).set_frame_rate(16000)
                emotion_input = AudioSegment.from_wav(emo_sample).set_frame_rate(16000)

                neutral_input += emotion_input.dBFS - neutral_input.dBFS
                combined_input = emotion_input + neutral_input

                out_name = os.path.join(combine_path, emo_sample.split("/")[-1][:-4] + "_" + neutral_sample.split("/")[-1])
                combined_input.export(out_name, format="wav")

                data_json = {
                    "wav": out_name,
                    "duration": len(combined_input) / 1000,
                    "emotion": [{
                        "emo": emotion,
                        "start": 0,
                        "end": len(emotion_input) / 1000,
                    }],
                }

                audio, audio_length = utils.load_wav(out_name)
                utterance_map['audio'] = audio
                utterance_map['audio_length'] = get_feat_extract_output_lengths(audio_length)
                utterance_map['text'] = " ".join([emotion_list[0]['text'], neutral_list[0]['text']])
                utterance_map['tlabel'] = tokenize(utterance_map['text'])
                utterance_map['frame_tlabel'] = [emotion] * (len(emotion_list[0]['text']) + 1) + [3] * (len(neutral_list[0]['text']) + 1)
                utterance_map['label'], utterance_map['frame_label'] = get_labels(data_json, utterance_map['audio_length'])
                utterance_map['frame_tlabel_length'] = getLength(utterance_map['frame_tlabel'])
                utterance_map['frame_label_length'] = getLength(utterance_map['frame_label'])

                emotion_list = emotion_list[1:]
                neutral_list = neutral_list[1:]

            elif combination == "neu_emo_neu":
                neutral_sample_1 = neutral_list[0]['audio']
                neutral_sample_2 = neutral_list[1]['audio']
                emo_sample = emotion_list[0]['audio']
                emotion = emotion_list[0]['label']

                neutral_input_1 = AudioSegment.from_wav(neutral_sample_1).set_frame_rate(16000)
                neutral_input_2 = AudioSegment.from_wav(neutral_sample_2).set_frame_rate(16000)
                emotion_input = AudioSegment.from_wav(emo_sample).set_frame_rate(16000)

                emotion_input += neutral_input_1.dBFS - emotion_input.dBFS
                neutral_input_2 += neutral_input_1.dBFS - neutral_input_2.dBFS
                combined_input = (neutral_input_1 + emotion_input + neutral_input_2)

                out_name = os.path.join(
                    combine_path,
                    neutral_sample_1.split("/")[-1][:-4] + "_" + emo_sample.split("/")[-1][:-4] + "_" + neutral_sample_2.split("/")[-1])
                combined_input.export(out_name, format="wav")

                data_json = {
                    "wav":
                    out_name,
                    "duration":
                    len(combined_input) / 1000,
                    "emotion": [{
                        "emo": emotion,
                        "start": len(neutral_input_1) / 1000,
                        "end": len(neutral_input_1) / 1000 + len(emotion_input) / 1000,
                    }],
                }

                audio, audio_length = utils.load_wav(out_name)
                utterance_map['audio'] = audio
                utterance_map['audio_length'] = get_feat_extract_output_lengths(audio_length)
                utterance_map['text'] = " ".join([neutral_list[0]['text'], emotion_list[0]['text'], neutral_list[1]['text']])
                utterance_map['tlabel'] = tokenize(utterance_map['text'])
                utterance_map['frame_tlabel'] = [3] * (len(neutral_list[0]['text']) + 2) + [emotion] * len(
                    emotion_list[0]['text']) + [3] * (len(neutral_list[1]['text']) + 1)
                utterance_map['label'], utterance_map['frame_label'] = get_labels(data_json, utterance_map['audio_length'])
                utterance_map['frame_tlabel_length'] = getLength(utterance_map['frame_tlabel'])
                utterance_map['frame_label_length'] = getLength(utterance_map['frame_label'])

                emotion_list = emotion_list[1:]
                neutral_list = neutral_list[2:]

            else:
                emo_sample_1 = emotion_list[0]['audio']
                emotion = emotion_list[0]['label']

                emotion_input_1 = AudioSegment.from_wav(emo_sample_1).set_frame_rate(16000)

                out_name = os.path.join(combine_path, emo_sample_1.split("/")[-1])
                emotion_input_1.export(out_name, format="wav")

                data_json = {
                    "wav": out_name,
                    "duration": len(emotion_input_1) / 1000,
                    "emotion": [{
                        "emo": emotion,
                        "start": 0,
                        "end": len(emotion_input_1) / 1000,
                    }],
                }

                audio, audio_length = utils.load_wav(out_name)
                utterance_map['audio'] = audio
                utterance_map['audio_length'] = get_feat_extract_output_lengths(audio_length)
                utterance_map['text'] = emotion_list[0]['text']
                utterance_map['tlabel'] = tokenize(utterance_map['text'])
                utterance_map['frame_tlabel'] = [emotion] * (len(emotion_list[0]['text']) + 1)
                utterance_map['label'], utterance_map['frame_label'] = get_labels(data_json, utterance_map['audio_length'])
                utterance_map['frame_tlabel_length'] = getLength(utterance_map['frame_tlabel'])
                utterance_map['frame_label_length'] = getLength(utterance_map['frame_label'])

                emotion_list = emotion_list[1:]
            assert len(utterance_map['frame_tlabel_length']) == len(utterance_map['label']) and len(
                utterance_map['frame_label_length']) == len(utterance_map['label']), "Unequal"
            CombList.append(utterance_map)

    with open(out_pikdir, 'wb') as f:
        pkl.dump(CombList, f)


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    nowdir = args.prodir
    raw_pikdir = f'{nowdir}/{dataset}_raw.pkl'
    combdir = f"./{dataset}_combined"
    out_pikdir = f'{nowdir}/{dataset}_mix.pkl'
    symbol_table = read_symbol_table(f'{nowdir}/char_units.txt', mode='char')
    ConcatRaw(combdir, raw_pikdir, out_pikdir, dataset)
