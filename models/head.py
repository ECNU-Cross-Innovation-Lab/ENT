import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_padding_mask


class EmoHead(nn.Module):
    def __init__(self, enc_dim, pred_dim, mode='pool', num_classes=4):
        super().__init__()
        self.mode = mode
        self.head = nn.Linear(enc_dim + pred_dim, num_classes)
        self.norm = nn.LayerNorm(enc_dim + pred_dim)

    def forward(self, enc_out, audio_length, pred_out, text_length):
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            audio_length (torch.Tensor): [B]
            pred_out (torch.Tensor): [B, U, P]
            text_length (torch.Tensor): [B]
        Return:
            [B, n_cls]
        """
        B, T, E = enc_out.shape
        B, U, P = pred_out.shape
        if self.mode == 'pool': 
            enc_mask = get_padding_mask(T, B, audio_length).unsqueeze(-1).expand(-1, -1, E).bool()  # [B, T, E]
            enc_out = enc_out.masked_fill(~enc_mask, 0)
            enc_out = torch.sum(enc_out, dim=1) / audio_length.unsqueeze(-1)  # [B, E]
            
            pred_mask = get_padding_mask(U, B, text_length).unsqueeze(-1).expand(-1, -1, P).bool()  # [B, U, P]
            pred_out = pred_out.masked_fill(~pred_mask, 0)
            pred_out = torch.sum(pred_out, dim=1) / text_length.unsqueeze(-1)  # [B, P]
        elif self.mode == 'last':
            enc_index = torch.sub(audio_length, 1).view(B, 1, 1).expand(-1, -1, E) # [B, 1, E]
            enc_out = torch.gather(enc_out, 1, enc_index).squeeze(1)
            pred_index = torch.sub(text_length, 1).view(B, 1, 1).expand(-1, -1, P) # [B, 1, P]
            pred_out = torch.gather(pred_out, 1, pred_index).squeeze(1)
        out = self.head(self.norm(torch.cat((enc_out, pred_out), dim=-1)))
        # out = self.head(torch.cat((enc_out, pred_out), dim=-1))
        return out
    

class LmHead(nn.Module):
    def __init__(self, pred_dim, vocab_size):
        super().__init__()
        self.head = nn.Linear(pred_dim, vocab_size)

    def forward(self, pred_out):
        """
        Args:
            pred_out (torch.Tensor): [B, U, P]
            text_length (torch.Tensor): [B]
        Return:
            [B, U, n_vocab]
        """
        B, U, P = pred_out.shape
        out = self.head(pred_out)
        return out


class EmoHeadMix(nn.Module):
    def __init__(self, enc_dim, pred_dim, mode='pool', num_classes=4):
        super().__init__()
        self.mode = mode
        self.head = nn.Linear(enc_dim + pred_dim, num_classes)
        self.norm = nn.LayerNorm(enc_dim + pred_dim)

    def forward(self, enc_out, pred_out, frame_label, frame_tlabel, label):
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, U, P]
            frame_label (torch.Tensor): [B, T]
            frame_tlabel (torch.Tensor): [B, U]
            label (List[List[int]]): [B,*]
        Return:
            [B, n_cls]
        """
        B, T, E = enc_out.shape
        B, U, P = pred_out.shape
        loss = 0
        cnt = 0
        for i in range(B):
            frame_label_span = frame_label[i] # [T]
            frame_tlabel_span = frame_tlabel[i] # [U]
            label_span = label[i]
            for emo in label_span:
                enc_out_span_emo = torch.mean(enc_out[i][frame_label_span.eq(emo), :],dim=0) # [E]
                pred_out_span_emo = torch.mean(pred_out[i][frame_tlabel_span.eq(emo), :],dim=0) # [P]
                out = self.head(self.norm(torch.cat((enc_out_span_emo, pred_out_span_emo), dim=-1)))
                loss += -F.log_softmax(out,dim=-1)[emo]
                cnt += 1
        loss /= cnt
        return loss


