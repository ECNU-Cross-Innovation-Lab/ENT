import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertModel, HubertConfig, WavLMModel, WavLMConfig, AutoModel
from .utils import ScalarMix


class Speech_Pretrain_Model(nn.Module):
    def __init__(self, use_emotion=False, pretrain='hubert', finetune=False) -> None:
        super().__init__()
        self.finetune = finetune
        self.use_emotion = use_emotion
        configmap = {'mask_time_prob': 0.08, 'mask_time_length': 15, 'mask_feature_prob': 0.05, 'mask_feature_length': 64}
        assert pretrain in ['hubert', 'wav2vec2', 'wavlm'], "Unkown pretrain model for finetuning"

        self.finetune = finetune
        self.epoch = 0

        if pretrain == 'hubert':
            config = HubertConfig.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=~finetune)
            config.update(configmap)
            self.pretrain = HubertModel.from_pretrained("facebook/hubert-base-ls960", config=config)

        elif pretrain == 'wav2vec2':
            config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base", output_hidden_states=~finetune)
            config.update(configmap)
            self.pretrain = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", config=config)
            
        elif pretrain == 'wavlm':
            config = WavLMConfig.from_pretrained("microsoft/wavlm-base", output_hidden_states=~finetune)
            config.update(configmap)
            self.pretrain = WavLMModel.from_pretrained("microsoft/wavlm-base", config=config)

        if not finetune:
            self.weight_audio = ScalarMix(13)
            if self.use_emotion:
                self.weight_emotion = ScalarMix(13)
        else:
            self.pretrain.freeze_feature_encoder()
            self.freeze_epoch = 0

    def set_num_updates(self):
        self.epoch += 1

    def forward(self, x):
        '''
        x : seq_num, seq_len
        '''

        if self.finetune:
            x_emotion = None
            ft = self.freeze_epoch <= self.epoch
            with torch.no_grad() if not ft else contextlib.ExitStack():
                x_audio = self.pretrain(x).last_hidden_state
        else:
            self.pretrain.eval()
            with torch.no_grad():
                x = self.pretrain(x).hidden_states
            x_audio = self.weight_audio(x)
            x_emotion = self.weight_emotion(x) if self.use_emotion else None
        return x_audio, x_emotion
