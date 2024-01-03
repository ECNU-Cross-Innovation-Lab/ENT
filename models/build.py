from .encoder import RNNEncoder
from .predictor import RNNPredictor
from .joint import Joint
from .transducer import ENT, FENT
from .head import EmoHead, LmHead


def build_model(config):
    model_type = config.MODEL.TYPE
    if config.MODEL.HEAD_WEIGHT:
        head_emotion = EmoHead(enc_dim=config.MODEL.ENCODER.ENC_DIM,
                               pred_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                               num_classes=config.MODEL.NUM_CLASSES)
    else:
        head_emotion = None
    if config.MODEL.JOINT_WEIGHT:
        joint_emotion = Joint(vocab_size=config.MODEL.NUM_CLASSES,
                              enc_dim=config.MODEL.ENCODER.ENC_DIM,
                              pred_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                              join_dim=config.MODEL.JOINT.JOIN_DIM,
                              joint_mode=config.MODEL.JOINT.JOINT_MODE,
                              pre_join=config.MODEL.JOINT.PRE_JOIN,
                              post_join=config.MODEL.JOINT.POST_JOIN)
    else:
        joint_emotion = None
    if config.MODEL.LM_WEIGHT:
        head_lm = LmHead(pred_dim=config.MODEL.PREDICTOR.OUTPUT_DIM, vocab_size=config.MODEL.VOCAB_SIZE)
    else:
        head_lm = None

    use_emotion = True if (model_type == 'fent') and (config.MODEL.JOINT_WEIGHT or config.MODEL.HEAD_WEIGHT) else False

    encoder = RNNEncoder(enc_dim=config.MODEL.ENCODER.ENC_DIM,
                         use_emotion=use_emotion,
                         drop=config.MODEL.DROP_RATE,
                         pretrain=config.MODEL.PRETRAIN,
                         finetune=config.TRAIN.FINETUNE)

    if model_type == 'ent':
        predictor = RNNPredictor(vocab_size=config.MODEL.VOCAB_SIZE,
                                 embed_dim=config.MODEL.PREDICTOR.EMBED_DIM,
                                 hidden_dim=config.MODEL.PREDICTOR.HIDDEN_DIM,
                                 output_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                                 drop=config.MODEL.DROP_RATE)
        joint = Joint(vocab_size=config.MODEL.VOCAB_SIZE,
                      enc_dim=config.MODEL.ENCODER.ENC_DIM,
                      pred_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                      join_dim=config.MODEL.JOINT.JOIN_DIM,
                      joint_mode=config.MODEL.JOINT.JOINT_MODE,
                      pre_join=config.MODEL.JOINT.PRE_JOIN,
                      post_join=config.MODEL.JOINT.POST_JOIN)

        model = ENT(vocab_size=config.MODEL.VOCAB_SIZE,
                    encoder=encoder,
                    predictor=predictor,
                    joint=joint,
                    joint_emotion=joint_emotion,
                    head_emotion=head_emotion,
                    head_lm=head_lm,
                    lm_weight=config.MODEL.LM_WEIGHT,
                    joint_weight=config.MODEL.JOINT_WEIGHT,
                    head_weight=config.MODEL.HEAD_WEIGHT,
                    rnnt_weight=config.MODEL.RNNT_WEIGHT)
    elif model_type == 'fent':
        predictor_blank = RNNPredictor(vocab_size=config.MODEL.VOCAB_SIZE,
                                       embed_dim=config.MODEL.PREDICTOR.EMBED_DIM,
                                       hidden_dim=config.MODEL.PREDICTOR.HIDDEN_DIM,
                                       output_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                                       drop=config.MODEL.DROP_RATE)
        predictor_vocab = RNNPredictor(vocab_size=config.MODEL.VOCAB_SIZE,
                                       embed_dim=config.MODEL.PREDICTOR.EMBED_DIM,
                                       hidden_dim=config.MODEL.PREDICTOR.HIDDEN_DIM,
                                       output_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                                       drop=config.MODEL.DROP_RATE)
        joint_blank = Joint(vocab_size=1,
                            enc_dim=config.MODEL.ENCODER.ENC_DIM,
                            pred_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                            join_dim=config.MODEL.JOINT.JOIN_DIM,
                            joint_mode=config.MODEL.JOINT.JOINT_MODE,
                            pre_join=config.MODEL.JOINT.PRE_JOIN,
                            post_join=config.MODEL.JOINT.POST_JOIN)
        joint_vocab = Joint(vocab_size=config.MODEL.VOCAB_SIZE - 1,
                            enc_dim=config.MODEL.ENCODER.ENC_DIM,
                            pred_dim=config.MODEL.PREDICTOR.OUTPUT_DIM,
                            join_dim=config.MODEL.JOINT.JOIN_DIM,
                            joint_mode=config.MODEL.JOINT.JOINT_MODE,
                            pre_join=config.MODEL.JOINT.PRE_JOIN,
                            post_join=config.MODEL.JOINT.POST_JOIN)
        model = FENT(vocab_size=config.MODEL.VOCAB_SIZE,
                     encoder=encoder,
                     predictor_blank=predictor_blank,
                     joint_blank=joint_blank,
                     predictor_vocab=predictor_vocab,
                     joint_vocab=joint_vocab,
                     joint_emotion=joint_emotion,
                     head_emotion=head_emotion,
                     head_lm=head_lm,
                     lm_weight=config.MODEL.LM_WEIGHT,
                     joint_weight=config.MODEL.JOINT_WEIGHT,
                     head_weight=config.MODEL.HEAD_WEIGHT,
                     rnnt_weight=config.MODEL.RNNT_WEIGHT)

    return model
