import os
from pickle import FALSE
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
# Path of log
_C.CFGFILE = './configs'
_C.LOGPATH = './log'
# Training mode ['asr', 'asrser']
_C.MODE = 'asrser'
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 2
# Dataset name
_C.DATA.DATASET = 'IEMOCAP'
# Path to dataset, overwritten by funcition ConfigDataset
_C.DATA.DATA_PATH = ''
# Path to zed dataset
_C.DATA.ZED_PATH = './dataset/ZED/zed.pkl'
# Whether to test on emotion diarization
_C.DATA.SED = False
# Tokenizer mode
_C.DATA.BPE_MODE = 'char'
# Path to symbol table
_C.DATA.SYMBOL_PATH = ''
# Path to BPE model
_C.DATA.BPEMODEL_PATH = ''
# Feature augmentation
_C.DATA.SPEAUG = True
# Sequence Length of pretrained speech representation
_C.DATA.LENGTH = 374
# Sequence Length of raw audio
_C.DATA.AUDIO_MAX_LENGTH = 120000
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type :['ent', 'fent']
_C.MODEL.TYPE = 'ent'
# Model name
_C.MODEL.NAME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 4
# Size of vocabulary, including <blank>
_C.MODEL.VOCAB_SIZE = 2830
# Pretrained Model in ['hubert','wav2vec2']
_C.MODEL.PRETRAIN = 'wav2vec2'
# Dropout rate
_C.MODEL.DROP_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0
# Loss type of emotion head
_C.MODEL.HEAD_LOSS = 'ce'
# Weight of emotion head loss, 0 for no use
_C.MODEL.HEAD_WEIGHT = 0.5
# Loss type of joint emotion detector
_C.MODEL.JOINT_LOSS = 'ce'
# Weight of fine-grained joint loss, 0 for no use
_C.MODEL.JOINT_WEIGHT = 0.5
# Weight of language modeling loss
_C.MODEL.LM_WEIGHT = 0.0
# Weight of RNNT loss
_C.MODEL.RNNT_WEIGHT = 0.5
# Whether to use temporal shift
_C.MODEL.USE_SHIFT = False
# kernel size of convolution
_C.MODEL.KERNEL_SIZE = 7
# path to save model
_C.MODEL.SAVE_PATH = './pth'
# whether to save model
_C.MODEL.SAVE = False

# Acoustic encoder parameters
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.ENC_DIM = 768

# Predictor parameters
_C.MODEL.PREDICTOR = CN()
_C.MODEL.PREDICTOR.EMBED_DIM = 256
_C.MODEL.PREDICTOR.HIDDEN_DIM = 256
_C.MODEL.PREDICTOR.OUTPUT_DIM = 256

# Joint network parameters
_C.MODEL.JOINT = CN()
# Only works when PRE_JOIN is True
_C.MODEL.JOINT.JOIN_DIM = 1024
_C.MODEL.JOINT.JOINT_MODE = 'add'
_C.MODEL.JOINT.PRE_JOIN = False
_C.MODEL.JOINT.POST_JOIN = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 10
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# whether to finetune or feature extraction
_C.TRAIN.FINETUNE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.SEED = 42
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# fold validation
_C.NUM_FOLD = 5


def Update(config, args):
    config.defrost()
    if args.model:
        config.MODEL.TYPE = args.model
    if args.sed:
        config.DATA.SED = True
    if args.bpe_mode:
        config.DATA.BPE_MODE = args.bpe_mode
    if args.finetune:
        config.TRAIN.FINETUNE = True
        config.CFGFILE = os.path.join(config.CFGFILE, config.DATA.BPE_MODE, '{}_{}_{}_finetune.yaml'.format(config.DATA.DATASET, config.MODEL.TYPE, config.MODE))
    else:
        config.TRAIN.FINETUNE = False
        config.CFGFILE = os.path.join(config.CFGFILE, 'sed' if config.DATA.SED else config.DATA.BPE_MODE, '{}_{}_{}.yaml'.format(config.DATA.DATASET, config.MODEL.TYPE, config.MODE))
    config.merge_from_file(config.CFGFILE)
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.symbol_table:
        config.DATA.SYMBOL_PATH = args.symbol_table
    if args.bpe_model:
        config.DATA.BPEMODEL_PATH = args.bpe_model
    if args.batchsize:
        config.DATA.BATCH_SIZE = args.batchsize
    if args.rnntw:
        config.MODEL.RNNT_WEIGHT = args.rnntw
    if args.lmw:
        config.MODEL.LM_WEIGHT = args.lmw
    if args.jointw:
        config.MODEL.JOINT_WEIGHT = args.jointw
    if args.headw:
        config.MODEL.HEAD_WEIGHT = args.headw
    if args.gpu:
        config.LOCAL_RANK = int(args.gpu)
    if args.seed:
        config.SEED = args.seed
    config.freeze()


def Rename(config):
    config.defrost()
    if config.MODEL.NAME == '':
        # config.MODEL.NAME = config.MODEL.TYPE
        config.MODEL.NAME = config.MODEL.TYPE + f'-hw{config.MODEL.HEAD_WEIGHT}-jw{config.MODEL.JOINT_WEIGHT}-rw{config.MODEL.RNNT_WEIGHT}'
    if config.DATA.SED:
        config.MODEL.NAME = 'sed-' + config.MODEL.NAME
    config.MODEL.NAME = config.MODEL.NAME + '-' + config.MODE
    
    if config.TRAIN.FINETUNE:
        config.MODEL.NAME = config.MODEL.NAME + '_finetune' + config.MODEL.PRETRAIN
    else:
        config.MODEL.NAME = config.MODEL.NAME + '_featurex' + config.MODEL.PRETRAIN
    config.MODEL.NAME = config.MODEL.NAME + '-' + config.DATA.DATA_PATH.split('/')[-1].split('.pkl')[0]
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    config = _C.clone()
    # ConfigDataset(config)
    Update(config, args)
    # ConfigPretrain(config)
    Rename(config)
    return config
