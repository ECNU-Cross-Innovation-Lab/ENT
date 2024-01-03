import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import unravel_index, get_interval_mask


class LatticeLoss(nn.Module):
    def __init__(self, mode = 'One'):
        '''
        Args:
            mode (str): Strategy for selecting the most emotional/non-emotional node.
                'One': for the single node
                'FullT': for all the nodes within the timestamp where the most emotional/non-emotional node is located, namely the entire row.
                'FullU': for all the nodes within the token where the most emotional/non-emotional node is located, namely the entire column.
        '''
        super().__init__()
        assert mode in ['One', 'FullT', 'FullU'], "Unkown lattice loss mode"
        self.mode = mode

    def forward(self, logits, label, mask):
        '''
        logits: B, T, U, class_num
        label: B, T, U
        mask: B, T, U
        '''
        loss = 0.0
        B, T, U = label.shape
        probs = F.softmax(logits, dim=-1)
        for i in range(B):
            label_index = label[i][0][0].item()
            prob = probs[i, :, :, label_index].masked_fill(~mask[i], 0.0) # target emotion probablity
            prob = torch.clamp(prob, 1e-8, 1.0)
            probn = probs[i, :, :, 3].masked_fill(~mask[i], 1.0) # neutral probablity
            probn = torch.clamp(probn, 1e-8, 1.0)
            if self.mode == 'One':
                max_prob = prob.max()
                loss += -torch.log(max_prob)
                min_probn = probn.min()
                loss += -torch.log(min_probn)
            else:
                max_prob_indice = unravel_index(torch.argmax(prob), prob.shape) # index max probability
                min_probn_indice = unravel_index(torch.argmin(probn), probn.shape) # index min probability
                if self.mode == 'FullT':
                    max_prob = prob[max_prob_indice[0], :]
                    min_probn = probn[min_probn_indice[0], :]
                else:
                    max_prob = prob[:, max_prob_indice[1]]
                    min_probn = probn[:, min_probn_indice[1]]
                loss += -torch.log(max_prob).mean() - torch.log(min_probn).mean()

        return loss / B


class LatticeLossMix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, label, frame_label_length, frame_tlabel_length):
        '''
        logits: B, T, U, class_num
        label: B*List [int]
        frame_label_length: B*List [int]
        frame_tlabel_length: B*List [int]
        '''
        loss = 0.0
        B, T, U, _ = logits.shape
        probs = F.softmax(logits, dim=-1)
        for i in range(B):
            prob_lattice = probs[i] # [T, U, n_cls]
            t_label = frame_label_length[i]
            u_label = frame_tlabel_length[i]
            t_start = 0
            u_start = 0
            for index, emo in enumerate(label[i]):
                t_mask = get_interval_mask(T, t_start, t_start+t_label[index], prob_lattice.device).unsqueeze(-1).expand(-1,U) # [T, U]
                u_mask = get_interval_mask(U, u_start, u_start+u_label[index], prob_lattice.device).unsqueeze(0).expand(T,-1)
                mask_tu = torch.logical_or(t_mask, u_mask)
                t_start += t_label[index]
                u_start += u_label[index]
                t_mask_whole = get_interval_mask(T, 0, t_start, prob_lattice.device).unsqueeze(-1).expand(-1,U) # [T, U]
                u_mask_whole = get_interval_mask(U, 0, u_start, prob_lattice.device).unsqueeze(0).expand(T,-1)
                mask_whole = torch.logical_and(t_mask_whole, u_mask_whole)
                mask = torch.logical_and(mask_tu, mask_whole) # [T, U]
                prob = prob_lattice[mask,:] # [N , n_cls]
                prob = torch.clamp(prob, 1e-8, 1.0)
                loss += -torch.log(prob[:,emo]).mean()

        return loss / B
