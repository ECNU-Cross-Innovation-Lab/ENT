import torch
import torchaudio
import torch.nn.functional as F
from torch import nn
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .loss import LatticeLoss
from .utils import add_blank, get_padding_mask
from .greedy_search import rnnt_greedy_search, frnnt_greedy_search


@dataclass
class ClsOutput(ModelOutput):
    loss: torch.FloatTensor = None
    loss_rnnt: Optional[torch.FloatTensor] = None
    head_logits: Optional[torch.FloatTensor] = None


class ENT(nn.Module):
    def __init__(self,
                 vocab_size,
                 encoder,
                 predictor,
                 joint,
                 joint_emotion=None,
                 head_emotion=None,
                 joint_weight=0.0,
                 head_weight=0.5,
                 lm_weight=0.0,
                 head_lm=None,
                 rnnt_weight=0.5,
                 blank=0,
                 ignore_id=-1):
        super().__init__()

        self.blank = blank
        self.ignore_id = ignore_id
        self.vocab_size = vocab_size
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.head_weight = head_weight
        self.joint_weight = joint_weight
        self.lm_weight = lm_weight
        self.rnnt_weight = rnnt_weight
        self.use_emotion = True if joint_weight or head_weight else False
        self.finetune = self.encoder.finetune
        if self.joint_weight:
            self.joint_emotion = joint_emotion
            self.joint_emofn = LatticeLoss()
        if self.head_weight:
            self.head_emotion = head_emotion
        if self.lm_weight:
            self.head_lm = head_lm

    def forward(self, audio, audio_length, text, text_length, label):
        """
        Args:
            audio: (Batch, Length, ...)
            audio_length: (Batch, )
            text: (Batch, Length)
            text_length: (Batch,)
        """

        # encoder
        enc_out, _ = self.encoder(audio, audio_length)

        # predictor
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        pred_out = self.predictor(ys_in_pad)

        # joint
        joint_out = self.joint(enc_out, pred_out)  # [B,T,U,V]
        padded_text_length = torch.add(text_length, 1)

        if self.head_weight:
            head_logits = self.head_emotion(enc_out, audio_length, pred_out, padded_text_length)
            loss_emohead = self.head_weight * F.cross_entropy(head_logits, label)
        else:
            head_logits = None
            loss_emohead = 0

        if self.joint_weight:
            joint_logits = self.joint_emotion(enc_out, pred_out)
            B, T, U, _ = joint_logits.shape
            joint_label = label.view(B, 1, 1).expand(-1, T, U)  # [B, T, U]
            enc_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, U).bool()  # [B, T, U]
            pred_mask = get_padding_mask(U, B, padded_text_length).unsqueeze(1).expand(-1, T, -1).bool()  # [B, T, U]
            mask = torch.logical_and(enc_mask, pred_mask)
            # loss_emojoint = self.joint_weight * F.cross_entropy(joint_logits[mask], joint_label[mask])
            loss_emojoint = self.joint_weight * self.joint_emofn(joint_logits, joint_label, mask)
        else:
            joint_logits = None
            loss_emojoint = 0

        if self.training and self.lm_weight:
            lm_logits = self.head_lm(pred_out)
            B, U, V = lm_logits.shape
            mask = get_padding_mask(U, B, padded_text_length)
            padded_text = F.pad(text, [0, 1], "constant", self.ignore_id)
            padded_text = torch.scatter(padded_text, 1, text_length.unsqueeze(-1), self.eos)
            loss_lm = self.lm_weight * F.cross_entropy(lm_logits[mask], padded_text[mask])
        else:
            loss_lm = 0

        # rnnt-loss
        if self.training:
            rnnt_text = torch.where(text == self.ignore_id, 0, text).to(torch.int32)
            rnnt_text_length = text_length.to(torch.int32)
            audio_length = audio_length.to(torch.int32)
            loss_rnnt = self.rnnt_weight * torchaudio.functional.rnnt_loss(
                joint_out, rnnt_text, audio_length, rnnt_text_length, blank=self.blank, reduction="mean")
        else:
            loss_rnnt = 0
        return ClsOutput(loss=loss_rnnt + loss_emohead + loss_emojoint + loss_lm, head_logits=head_logits)

    def greedy_search(
        self,
        audio: torch.Tensor,
        audio_length: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        n_steps: int = 32,
    ):
        """ greedy search

        Args:
            audio (torch.Tensor): (batch=1, max_len, feat_dim)
            audio_length (torch.Tensor): (batch, )
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        _ = simulate_streaming
        enc_out, _ = self.encoder(audio, audio_length)
        hyps = rnnt_greedy_search(self, enc_out, audio_length, n_steps=n_steps)
        # hyps = rnnt_beam_search(self, enc_out, audio_length, n_steps=n_steps)

        return hyps


class FENT(nn.Module):
    def __init__(self,
                 vocab_size,
                 encoder,
                 predictor_blank,
                 joint_blank,
                 predictor_vocab,
                 joint_vocab,
                 joint_emotion=None,
                 joint_weight=0.5,
                 head_emotion=None,
                 head_weight=0.5,
                 lm_weight=0.0,
                 head_lm=None,
                 rnnt_weight=0.5,
                 blank=0,
                 ignore_id=-1):
        super().__init__()

        self.blank = blank
        self.ignore_id = ignore_id
        self.vocab_size = vocab_size
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.encoder = encoder
        self.predictor_blank = predictor_blank
        self.joint_blank = joint_blank
        self.predictor_vocab = predictor_vocab
        self.joint_vocab = joint_vocab
        self.head_weight = head_weight
        self.joint_weight = joint_weight
        self.lm_weight = lm_weight
        self.rnnt_weight = rnnt_weight
        self.use_emotion = True if joint_weight or head_weight else False
        self.finetune = self.encoder.finetune
        if self.joint_weight:
            self.joint_emotion = joint_emotion
            self.joint_emofn = LatticeLoss()
        if self.head_weight:
            self.head_emotion = head_emotion
        if self.lm_weight:
            self.head_lm = head_lm

    def forward(self, audio, audio_length, text, text_length, label):
        """
        Args:
            audio: (Batch, Length, ...)
            audio_length: (Batch, )
            text: (Batch, Length)
            text_length: (Batch,)
        """

        # encoder
        enc_out, enc_emo_out = self.encoder(audio, audio_length)

        # predictor
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        predb_out = self.predictor_blank(ys_in_pad)
        predv_out = self.predictor_vocab(ys_in_pad)

        # joint
        jointb_out = self.joint_blank(enc_out if enc_emo_out is None else enc_emo_out, predb_out)  # [B,T,U,1]
        jointv_out = self.joint_vocab(enc_out, predv_out)  # [B,T,U,V-1]
        joint_out = torch.cat((jointb_out, jointv_out), dim=-1)
        padded_text_length = torch.add(text_length, 1)
        if self.head_weight:
            head_logits = self.head_emotion(enc_out if enc_emo_out is None else enc_emo_out, audio_length, predb_out, padded_text_length)
            loss_emohead = self.head_weight * F.cross_entropy(head_logits, label)
        else:
            head_logits = None
            loss_emohead = 0

        if self.joint_weight:
            joint_logits = self.joint_emotion(enc_out if enc_emo_out is None else enc_emo_out, predb_out)  # [B,T,U,4]
            B, T, U, _ = joint_logits.shape
            joint_label = label.view(B, 1, 1).expand(-1, T, U)  # [B, T, U]
            enc_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, U).bool()  # [B, T, U]
            pred_mask = get_padding_mask(U, B, padded_text_length).unsqueeze(1).expand(-1, T, -1).bool()  # [B, T, U]
            mask = torch.logical_and(enc_mask, pred_mask)
            # loss_emojoint = self.joint_weight * F.cross_entropy(joint_logits[mask], joint_label[mask])
            loss_emojoint = self.joint_weight * self.joint_emofn(joint_logits, joint_label, mask)
        else:
            joint_logits = None
            loss_emojoint = 0

        if self.training and self.lm_weight:
            lm_logits = self.head_lm(predv_out)
            B, U, V = lm_logits.shape
            mask = get_padding_mask(U, B, padded_text_length)
            padded_text = F.pad(text, [0, 1], "constant", self.ignore_id)
            padded_text = torch.scatter(padded_text, 1, text_length.unsqueeze(-1), self.eos)
            loss_lm = self.lm_weight * F.cross_entropy(lm_logits[mask], padded_text[mask])
        else:
            loss_lm = 0

        # rnnt-loss
        if self.training:
            rnnt_text = torch.where(text == self.ignore_id, 0, text).to(torch.int32)
            rnnt_text_length = text_length.to(torch.int32)
            audio_length = audio_length.to(torch.int32)
            loss_rnnt = self.rnnt_weight * torchaudio.functional.rnnt_loss(
                joint_out, rnnt_text, audio_length, rnnt_text_length, blank=self.blank, reduction="mean")
        else:
            loss_rnnt = 0
        return ClsOutput(loss=loss_rnnt + loss_emohead + loss_emojoint + loss_lm, head_logits=head_logits)

    def greedy_search(
        self,
        audio: torch.Tensor,
        audio_length: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        n_steps: int = 32,
    ):
        """ greedy search

        Args:
            audio (torch.Tensor): (batch=1, max_len, feat_dim)
            audio_length (torch.Tensor): (batch, )
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        _ = simulate_streaming
        enc_out, enc_emo_out = self.encoder(audio, audio_length)
        hyps = frnnt_greedy_search(self, enc_out, audio_length, enc_emo_out, n_steps=n_steps)
        # hyps = frnnt_beam_search(self, enc_out, audio_length, enc_emo_out, n_steps=n_steps)

        return hyps
