import torch
import torch.nn as nn
from typing import List


class RNNPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, drop=0.):
        super().__init__()
        self.n_layers = 1
        self.hidden_size = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(drop)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=self.n_layers, batch_first=True, dropout=drop)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, cache=None):
        """
        Args:
            x: batch_size, max_time
            cache: List[state_m, state_c]

        Returns:
            output: batch, max_time, output_size
        """
        embed = self.embed(x)
        embed = self.dropout(embed)
        if cache is None:
            state = self.init_state(batch_size=x.size(0), device=x.device)
            states = (state[0], state[1])
        else:
            states = (cache[0], cache[1])
        output, (m, c) = self.rnn(embed, states)
        output = self.projection(output)

        return output

    def init_state(self, batch_size: int, device):
        """
        Returns:
            cache (List[torch.Tensor])
                cache[*]: [num_layer, batch_size, hidden_dim]
        """

        return [
            torch.zeros(1 * self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(1 * self.n_layers, batch_size, self.hidden_size, device=device)
        ]

    def forward_step(self, input, padding, cache):
        """
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 0 is padding value
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
                    List[torch.Tensor[num_layer, B, hidden_dim]]
        """
        state_m, state_c = cache[0], cache[1]
        embed = self.embed(input)
        embed = self.dropout(embed)
        output, (now_m, now_c) = self.rnn(embed, (state_m, state_c))
        now_m = ApplyPadding(now_m, padding.unsqueeze(0), state_m)
        now_c = ApplyPadding(now_c, padding.unsqueeze(0), state_c)
        output = self.projection(output)
        return output, (now_m, now_c)
    
    def batch_to_cache(self,
                       cache: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        Args:
           cache: [state_m, state_c]
               state_ms: [1*n_layers, bs, ...]
               state_cs: [1*n_layers, bs, ...]
        Returns:
           new_cache: [[state_m_1, state_c_1], [state_m_2, state_c_2]...]
        """
        assert len(cache) == 2
        state_ms = cache[0]
        state_cs = cache[1]

        assert state_ms.size(1) == state_cs.size(1)

        new_cache: List[List[torch.Tensor]] = []
        for state_m, state_c in zip(torch.split(state_ms, 1, dim=1),
                                    torch.split(state_cs, 1, dim=1)):
            new_cache.append([state_m, state_c])
        return new_cache

    def cache_to_batch(self,
                       cache: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Args:
            cache : [[state_m_1, state_c_1], [state_m_1, state_c_1]...]

        Returns:
            new_caceh: [state_ms, state_cs],
                state_ms: [1*n_layers, bs, ...]
                state_cs: [1*n_layers, bs, ...]
        """
        state_ms = torch.cat([states[0] for states in cache], dim=1)
        state_cs = torch.cat([states[1] for states in cache], dim=1)
        return [state_ms, state_cs]


def ApplyPadding(input, padding, pad_value) -> torch.Tensor:
    """
    Args:
        input:   [bs, max_time_step, dim]
        padding: [bs, max_time_step]
    """
    return padding * pad_value + input * (1 - padding)
