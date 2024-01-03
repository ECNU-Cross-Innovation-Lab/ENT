import os
import re
import sys
import argparse
import torch
import torchaudio
import numpy as np
import pickle as pkl

import utils

emotion_id = {'hap': 0, 'ang': 1, 'sad': 2, 'neu': 3, 'exc': 0}
# second = 7.5

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="/Dataset/IEMOCAP", help='path of IEMOCAP dataset')
parser.add_argument('--prodir', type=str, default=".", help='directory of IEMOCAP process')


def generate_iemocap4(indir, pikdir, text_dir):
    """
    For this setting, the dataset is constructed by merging excited and happy while removing 'frustrated'.
    Number of class: 4
    """
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    annotator_line = re.compile(r'C-')
    SessionMap = {}
    file_text = open(text_dir, "w")
    for sess in range(1, 6):
        emo_evaluation_dir = os.path.join(indir, 'Session{}/dialog/EmoEvaluation/'.format(sess))
        transcription_dir = os.path.join(indir, 'Session{}/dialog/transcriptions/'.format(sess))
        emo_sentences_dir = os.path.join(indir, 'Session{}/sentences/wav/'.format(sess))
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        sess_name = 'Session{}'.format(sess)
        SessionMap[sess_name] = []
        print('Processing {}'.format(sess_name))
        for file in evaluation_files:
            with open(transcription_dir + file) as ft:
                text_content = ft.read()
            with open(emo_evaluation_dir + file) as f:
                line = f.readline()

                while line:

                    if re.match(info_line, line):  # each utterance begin
                        utterance_map = {}
                        emo_wav_dir = emo_sentences_dir + \
                            file.split('.')[0] + '/'
                        start_end_time, title_name, emotion, val_act_dom = line.strip().split('\t')
                        wav_file_name = emo_wav_dir + title_name + '.wav'
                        if emotion not in emotion_id.keys():
                            line = f.readline()
                            continue

                        # Process audio and label
                        audio, audio_length = utils.load_wav(wav_file_name)
                        utterance_map['audio'] = audio
                        utterance_map['audio_length'] = audio_length
                        utterance_map['label'] = emotion_id[emotion]

                        # Process text
                        pattern = r"{} \[.+\]: (.+)".format(title_name)
                        text_line = re.findall(pattern, text_content)[0]
                        text_line = text_line.upper()
                        punc = "\[\]\.\,\?\!\-\:\;\""
                        text_line = re.sub(r"[{0}]+".format(punc), " ", text_line)
                        utterance_map['text'] = " ".join(text_line.split())
                        file_text.write(utterance_map['text'] + '\n')

                        SessionMap[sess_name].append(utterance_map)
                    line = f.readline()
    file_text.close()
    with open(pikdir, 'wb') as f:
        pkl.dump(SessionMap, f)


if __name__ == '__main__':
    args = parser.parse_args()
    indir = args.path
    nowdir = args.prodir
    iemocap4_pikdir = f'{nowdir}/iemocap4.pkl'
    text_dir = f'{nowdir}/input.txt'
    # print(f'output pickle: {raw_pikdir}')
    generate_iemocap4(indir, iemocap4_pikdir, text_dir)
