import torch
import torch.nn as nn
from .pretrain import Speech_Pretrain_Model


class RNNEncoder(nn.Module):
    def __init__(self, enc_dim, use_emotion=False, drop=0., pretrain='wav2vec2', finetune=False):
        super().__init__()
        self.finetune = finetune
        self.use_emotion = use_emotion
        self.speech_pretrain_model = Speech_Pretrain_Model(use_emotion, pretrain, finetune)
        self.rnn = nn.LSTM(input_size=enc_dim, hidden_size=enc_dim//2, num_layers=1, batch_first=True, bidirectional=True, dropout=drop)

    def encode_with_rnn(self, x, length):
        total_length = x.size(1)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.rnn(rnn_inputs)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=total_length)
        return rnn_outputs

    def forward(self, x, length=None):
        """
        Args:
            x: batch_size, max_time
            length: batch_size
        
        Returns:
            rnn_output: batch_size, max_time, dim
        """
        if not self.finetune and self.use_emotion:
            x, y = self.speech_pretrain_model(x)  # [batch_size, max_time, dim]
            rnn_output = self.encode_with_rnn(x, length)
            rnn_emo_output = self.encode_with_rnn(y, length)
            return rnn_output, rnn_emo_output
        else:
            x, _ = self.speech_pretrain_model(x)  # [batch_size, max_time, dim]
            rnn_output = self.encode_with_rnn(x, length)
            return rnn_output, None
