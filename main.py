# The code architectue is adopted from https://github.com/microsoft/Swin-Transformer
import os
import sys
import logging
import argparse
import numpy as np
import sentencepiece as spm
from time import strftime, localtime
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from jiwer.measures import wer, cer

import torch
import torch.backends.cudnn as cudnn
# from timm.loss import LabelSmoothingCrossEntropy

from config import get_config
from models import build_model
from dataset import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from dataset.IEMOCAP.toke import read_symbol_table
from metrics import IEMOCAP_Meter, EDER

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

emo_dict = {0:'h', 1:'a', 2:'s', 3:'n'}

def parse_option():
    parser = argparse.ArgumentParser('ENT', add_help=False)

    # # easy config modification
    # parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--batchsize', type=int, help="batch size for single GPU")
    parser.add_argument('--symbol_table', type=str, help='path to symbol table')
    parser.add_argument('--bpe_model', type=str, help='path to BPE mode')
    parser.add_argument('--bpe_mode', type=str, choices=['bpe', 'char'], help='toekn encoding type')
    parser.add_argument('--model', type=str, choices=['ent', 'fent'], help='model type')
    parser.add_argument('--rnntw', type=float, help="weight of RNNT loss")
    parser.add_argument('--lmw', type=float, help="weight of language modeling")
    parser.add_argument('--jointw', type=float, help="weight of joint loss")
    parser.add_argument('--headw', type=float, help="weight of emotion head loss")
    parser.add_argument('--sed', action='store_true', help='whether to validate on emotion diariztion')
    parser.add_argument('--finetune', action='store_true', help='whether to finetune or feature extraction')
    parser.add_argument('--gpu', type=str, help='gpu rank to use')
    parser.add_argument('--seed', type=int, help='seed')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return config


def main(config):
    result = []
    if not config.DATA.SED:
        for ith_fold in range(1, config.NUM_FOLD + 1):
            WA, UA, WER, CER = solve(config, ith_fold)
            result.append([WA, UA, WER, CER])
        result = np.array(result)
        logger.info('#' * 30 + f'  Summary  ' + '#' * 30)
        logger.info('\n')
        logger.info('fold\tWER\tCER\t')
        for ith_fold in range(1, config.NUM_FOLD + 1):
            WER, CER = result[ith_fold - 1,2], result[ith_fold - 1,3]
            logger.info(f'{ith_fold}\t{WER:.2f}\t{CER:.2f}')
        WER_mean = np.mean(result[:, 2])
        CER_mean = np.mean(result[:, 3])
        logger.info('Avg_WER\tAvg_CER')
        logger.info(f'{WER_mean:.2f}\t{CER_mean:.2f}')
        logger.info('\n')

        logger.info('fold\tWA\tUA\t')
        for ith_fold in range(1, config.NUM_FOLD + 1):
            WA, UA = result[ith_fold - 1,0], result[ith_fold - 1,1]
            logger.info(f'{ith_fold}\t{WA:.2f}\t{UA:.2f}')
        WA_mean = np.mean(result[:, 0])
        UA_mean = np.mean(result[:, 1])
        logger.info('Avg_WA\tAvg_UA')
        logger.info(f'{WA_mean:.2f}\t{UA_mean:.2f}')
    else:
        solve(config)
    


def solve(config, ith_fold=1):
    dataloader_train, dataloader_test = build_loader(config, ith_fold=ith_fold)
    model = build_model(config)
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if ith_fold == 1:
        logger.info(str(model))
        logger.info(f"number of params: {n_parameters}")

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(dataloader_train))

    
    symbol_table = read_symbol_table(config.DATA.SYMBOL_PATH, config.DATA.BPE_MODE)
    if config.DATA.BPE_MODE == 'bpe':
        bpe_model = spm.SentencePieceProcessor()
        bpe_model.load(config.DATA.BPEMODEL_PATH)
    elif config.DATA.BPE_MODE == 'char':
        bpe_model = None


    # if config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()


    logger.info('#' * 30 + '  Start Training  ' + '#' * 30)
    Meter = IEMOCAP_Meter()

    for epoch in range(config.TRAIN.EPOCHS):
        logger.info(f'>> epoch {epoch}')
        train_loss = train_one_epoch(config, model, dataloader_train, optimizer, epoch, lr_scheduler)

        if not config.DATA.SED:
            # if epoch < 30: # For efficiency, you can skip initial epoches for evaluation
            #     continue
            test_loss, WA, UA, WER, CER = validate(config, dataloader_test, model, symbol_table, bpe_model)
            logger.info(f'train loss: {train_loss:.2f}, test loss: {test_loss:.2f}, WER: {WER:.2f}, CER: {CER:.2f}, WA: {WA:.2f}, UA: {UA:.2f}')
            if Meter.UA < UA and config.MODEL.SAVE:
                torch.save(model.state_dict(), f'{config.MODEL.SAVE_PATH}/{config.MODEL.NAME}_{ith_fold}.pth')
            Meter.update(WA, UA, WER, CER)
        else:
            test_loss, WA, UA, WER, CER, eder = validate_sed(config, dataloader_test, model, symbol_table, bpe_model)
            logger.info(f'train loss: {train_loss:.2f}, test loss: {test_loss:.2f}, WER: {WER:.2f}, CER: {CER:.2f}, WA: {WA:.2f}, UA: {UA:.2f}, EDER: {eder:.2f}')
            if Meter.WA < WA and config.MODEL.SAVE:
                torch.save(model.state_dict(), f'{config.MODEL.SAVE_PATH}/{config.MODEL.NAME}_WA.pth')
            if Meter.eder > eder and config.MODEL.SAVE:
                torch.save(model.state_dict(), f'{config.MODEL.SAVE_PATH}/{config.MODEL.NAME}_EDER.pth')
            Meter.update(WA, UA, WER, CER, eder)
    logger.info('#' * 30 + f'  Summary fold{ith_fold}  ' + '#' * 30)
    logger.info(f'MAX_WA: {Meter.WA:.2f}')
    logger.info(f'MAX_UA: {Meter.UA:.2f}')
    logger.info(f'MIN_WER: {Meter.WER:.2f}')
    logger.info(f'MIN_CER: {Meter.CER:.2f}')
    if config.DATA.SED:
        logger.info(f'MIN_EDER: {Meter.eder:.2f}')
    return Meter.WA, Meter.UA, Meter.WER, Meter.CER


def train_one_epoch(config, model, dataloader, optimizer, epoch, lr_scheduler):
    total_loss = 0
    optimizer.zero_grad()
    for i, (audio, audio_length, tlabel, tlabel_length, label, text) in enumerate(dataloader):
        audio = audio.cuda()
        audio_length = audio_length.cuda()
        tlabel = tlabel.cuda()
        tlabel_length = tlabel_length.cuda()
        label = label.cuda()
        # try:
        model.train()
        outputs = model(audio, audio_length, tlabel, tlabel_length, label)
        loss = outputs.loss
        num_steps = len(dataloader)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        optimizer.step()
        if not config.TRAIN.FINETUNE:
            lr_scheduler.step_update(epoch * num_steps + i)
        # except:
        #     print(f'Too long sequence length {audio.size(1)}')
        #     torch.cuda.empty_cache()
        
    if config.TRAIN.FINETUNE:
        lr_scheduler.step()
        model.encoder.speech_pretrain_model.set_num_updates()
    return total_loss


@torch.no_grad()
def validate(config, data_loader, model, symbol_table, bpe_model=None):
    # batch_size=1
    model.eval()
    total_loss = 0
    emo_pred_list = []
    emo_label_list = []
    text_pred_list = []
    text_label_list = []
    WA = 0
    UA = 0
    WER = 100
    CER = 100
    for idx, (audio, audio_length, tlabel, tlabel_length, label, text) in enumerate(data_loader):
        audio = audio.cuda()
        audio_length = audio_length.cuda()
        tlabel = tlabel.cuda()
        tlabel_length = tlabel_length.cuda()
        label = label.cuda()

        # search transcriptions
        hypos, emos = model.greedy_search(audio, audio_length)
        text_pred_list.append(hypos2str(hypos, config.DATA.BPE_MODE, symbol_table, bpe_model))
        text_label_list.extend(text)
        
        # get emotion and record loss
        try:
            if config.MODEL.HEAD_WEIGHT:
                tlabel = torch.tensor([hypos], device=audio.device)
                tlabel_length = torch.tensor([len(hypos)], device=audio.device)
                outputs = model(audio, audio_length, tlabel, tlabel_length, label)
                head_logits = outputs.head_logits
                emo_pred = list(torch.argmax(head_logits, 1).cpu().numpy())
                emo_label = list(label.cpu().numpy())
                emo_pred_list.extend(emo_pred)
                emo_label_list.extend(emo_label)
            else:
                outputs = model(audio, audio_length, tlabel, tlabel_length, label)
            loss = outputs.loss
            total_loss += loss.item()
        except:
            print(f'Too long sequence length {len(hypos)}') # Model in the initial few epochs perform poor in ASR, maybe producing longer output transcriptions.
        
    WER = wer(text_label_list, text_pred_list) * 100
    CER = cer(text_label_list, text_pred_list) * 100
    if len(emo_label_list):
        WA = accuracy_score(emo_label_list, emo_pred_list) * 100
        UA = balanced_accuracy_score(emo_label_list, emo_pred_list) * 100
    return total_loss, WA, UA, WER, CER


def hypos2str(hypos, mode, symbol_table, bpe_model=None):
    """
    Decode token ids into string.
    Args:
        hypos: List[int], list of token id
    """
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1
    piece_list = []
    for token_id in hypos:
        if token_id == eos:
            break
        piece_list.append(char_dict[token_id])
    if mode == 'bpe':
        text_str = bpe_model.decode(piece_list)
    else:
        text_str = ''.join(piece_list)
    return text_str
    
@torch.no_grad()
def validate_sed(config, data_loader, model, symbol_table, bpe_model=None):
    # batch_size=1
    model.eval()
    total_loss = 0
    eder_list = []
    emo_pred_list = []
    emo_label_list = []
    text_pred_list = []
    text_label_list = []
    WA = 0
    UA = 0
    eder = 100
    WER = 100
    CER = 100
    for idx, (id, audio, audio_length, duration, frame_label, interval, label, text) in enumerate(data_loader):
        audio = audio.cuda()
        audio_length = audio_length.cuda()
        label = label.cuda()

        # search transcriptions
        hypos, emos = model.greedy_search(audio, audio_length)
        emos = [emo_dict[emo] for emo in emos]
        
        text_pred_list.append(hypos2str(hypos, config.DATA.BPE_MODE, symbol_table, bpe_model))
        text_label_list.extend(text)
        eder_list.append(EDER(emos, id[0], duration[0], interval)*100)
        
        
    WER = wer(text_label_list, text_pred_list) * 100
    CER = cer(text_label_list, text_pred_list) * 100
    if len(emo_label_list):
        WA = accuracy_score(emo_label_list, emo_pred_list) * 100
        UA = balanced_accuracy_score(emo_label_list, emo_pred_list) * 100
    if len(eder_list):
        eder = np.mean(eder_list)
    return total_loss, WA, UA, WER, CER, eder


if __name__ == '__main__':
    config = parse_option()
    # log_file = '{}-{}-{}.log'.format(config.MODEL.NAME, config.DATA.DATASET, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    log_file = '{}-{}.log'.format(config.MODEL.NAME, config.DATA.DATASET)
    if not os.path.exists(config.LOGPATH):
        os.makedirs(config.LOGPATH)
    logger.addHandler(logging.FileHandler("%s/%s" % (config.LOGPATH, log_file)))
    logger.info('#' * 30 + '  Training Arguments  ' + '#' * 30)
    logger.info(config.dump())
    torch.cuda.set_device(config.LOCAL_RANK)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    main(config)
