import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List
from .utils import get_padding_mask

@dataclass
class ClsOutput(ModelOutput):
    loss: torch.FloatTensor = None
    loss_rnnt: Optional[torch.FloatTensor] = None
    head_logits: Optional[torch.FloatTensor] = None


class Classifier(nn.Module):
    def __init__(self, encoder, enc_dim, num_classes=4):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(enc_dim, num_classes)
        self.norm = nn.LayerNorm(enc_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.encoder.use_emotion = False
        self.encoder.speech_pretrain_model.use_emotion = False

    def forward(self, audio, audio_length, text, text_length, label):
        """
        Args:
            audio (torch.Tensor): [B, T]
            audio_length (torch.Tensor): [B]
        Return:
            [B, n_cls]
        """
        enc_out = self.encoder(audio, audio_length)
        B, T, E = enc_out.shape
        if self.training:
            enc_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, E).bool()  # [B, T, E]
            enc_out = enc_out.masked_fill(~enc_mask, 0)
            enc_out = torch.sum(enc_out, dim=1) / audio_length.unsqueeze(-1)  # [B, E]
            head_logits = self.head(self.norm(enc_out))
            loss = self.criterion(head_logits,label)
        else:
            head_logits = self.head(self.norm(enc_out))
            loss = 0
        return ClsOutput(loss=loss, head_logits=head_logits)