import torch
from torch.utils.data import DataLoader

import pickle
import numpy as np
from dataset.IEMOCAP import IEMOCAP
from dataset.ZED import ZED

def build_loader(config, ith_fold=1):
    dataset_train, dataset_test = build_dataset(config, ith_fold=ith_fold)
    dataloader_train = DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE, collate_fn=dataset_train.collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=1, collate_fn=dataset_test.collate_fn)
    return dataloader_train, dataloader_test


def build_dataset(config, ith_fold=1):
    with open(config.DATA.DATA_PATH,'rb') as f:
        IEMOCAP_DataMap = pickle.load(f)
    if config.DATA.SED:
        dataset_train = IEMOCAP.Merge(IEMOCAP_DataMap)
        with open(config.DATA.ZED_PATH,'rb') as f:
            ZED_DataMap = pickle.load(f)
        dataset_test = ZED.ZED(ZED_DataMap)
    elif config.DATA.DATASET == 'IEMOCAP':
        dataset_train, dataset_test = IEMOCAP.Partition(ith_fold, IEMOCAP_DataMap)

    return dataset_train, dataset_test

