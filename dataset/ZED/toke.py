import os
import re
import sys
import pickle
import argparse
import numpy as np
import pickle as pkl
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./iemocap4.pkl', help='path to dataset pickle')
parser.add_argument('--out_path', type=str, default='./iemocap4bpe.pkl', help='path to output dataset pickle')
parser.add_argument('--symbol_table', type=str, default='./iemocap_bpe5000_units.txt', help='path to symbol table')
parser.add_argument('--mode', type=str, default='bpe', help='char or bpe tokenizer')
parser.add_argument('--bpe_model', type=str, default='./iemocap_bpe5000.model', help='path to BPE model')


def read_symbol_table(symbol_table_file, mode='bpe'):
    # borrowed from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/file_utils.py
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            if mode == 'char' and arr[0] == '▁':
                symbol_table[' '] = int(arr[1])
            else:
                symbol_table[arr[0]] = int(arr[1])
    return symbol_table


def tokenize(data_path, out_path, mode, symbol_table_path, bpe_model_path):
    with open(data_path, 'rb') as f:
        DataMap = pickle.load(f)
    symbol_table = read_symbol_table(symbol_table_path, mode)
    if mode == 'bpe':
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model_path)

    for sess_name in DataMap.keys():
        for i, utterance_map in enumerate(DataMap[sess_name]):
            if mode == 'bpe':
                tokens = sp.encode_as_pieces(DataMap[sess_name][i]['text'])
            else:
                tokens = DataMap[sess_name][i]['text']
            tlabel = []
            for ch in tokens:
                if ch in symbol_table:
                    tlabel.append(symbol_table[ch])
                elif '<unk>' in symbol_table:
                    tlabel.append(symbol_table['<unk>'])
            # DataMap[sess_name][i]['token'] = tokens
            DataMap[sess_name][i]['tlabel'] = tlabel

    with open(out_path, 'wb') as f:
        pickle.dump(DataMap, f)


if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    out_path = args.out_path
    symbol_table_path = args.symbol_table
    bpe_model_path = args.bpe_model
    mode = args.mode
    tokenize(data_path, out_path, mode, symbol_table_path, bpe_model_path)