import torch
import torch.nn as nn


class Joint(nn.Module):
    def __init__(self, vocab_size, enc_dim, pred_dim, join_dim, joint_mode='add', pre_join=False, post_join=False):
        super().__init__()
        self.pre_join = pre_join
        self.post_join = post_join
        self.joint_mode = joint_mode
        if self.pre_join:
            self.enc_ffn = nn.Linear(enc_dim, join_dim)
            self.pred_ffn = nn.Linear(pred_dim, join_dim)
        else:
            if self.joint_mode == 'cat':
                join_dim = enc_dim + pred_dim
            else:
                assert enc_dim == pred_dim == join_dim, "Unmatched enc_dim, pred_dim, and join_dim"
                join_dim = enc_dim
            
        if self.post_join:
            self.post_ffn = nn.Linear(join_dim, join_dim)
        self.activation = nn.Tanh()
        self.ffn_out = nn.Linear(join_dim, vocab_size)

    def forward(self, enc_out, pred_out):
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, U, P]
        Return:
            [B,T,U,V]
        """
        T = enc_out.size(1)
        U = pred_out.size(1)
        if self.pre_join:
            enc_out = self.enc_ffn(enc_out)
            pred_out = self.pred_ffn(pred_out)
        enc_out = enc_out.unsqueeze(2).repeat(1, 1, U, 1)  # [B,T,E] -> [B,T,1,E] -> [B,T,U,E]
        pred_out = pred_out.unsqueeze(1).repeat(1, T, 1, 1)  # [B,U,P] -> [B,1,U,P] -> [B,T,U,P]
        if self.joint_mode == 'add':
            output = enc_out + pred_out
        elif self.joint_mode == 'cat':
            output = torch.cat((enc_out, pred_out), dim=-1)
        output = self.ffn_out(self.activation(output))

        return output
