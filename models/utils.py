import math
import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
from typing import List
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


class ScalarMix(nn.Module):
    """
    from https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    In addition, if `do_layer_norm=True` then apply layer normalization to each tensor
    before weighting.
    """
    def __init__(
        self,
        mixture_size: int,
        initial_scalar_parameters: List[float] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.mixture_size = mixture_size

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size

        assert len(initial_scalar_parameters
                   ) == self.mixture_size, "{} tensors were passed, but the module was initialized to mix {} tensors.".format(
                       len(initial_scalar_parameters), self.mixture_size)

        self.scalar_parameters = ParameterList(
            [Parameter(torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=trainable) for i in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.
        When `do_layer_norm=False` the `mask` is ignored.
        """
        assert len(tensors) == self.mixture_size, "{} tensors were passed, but the module was initialized to mix {} tensors.".format(
            len(tensors), self.mixture_size)

        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)


class Specaugment(nn.Module):
    """
    For convenience, we apply specaugment to the output of pretrained model.
    The code reference from https://github.com/b04901014/FT-w2v2-ser/blob/main/modules/FeatureFuser.py
    """
    def __init__(self, dim) -> None:
        super().__init__()
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(dim).uniform_())
        self.mask_time_length = 15
        self.mask_time_prob = 0.08
        self.observe_time_prob = 0.0
        self.mask_feature_length = 64
        self.mask_feature_prob = 0.05

    def forward(self, x):
        """
        x: B L D
        """
        if not self.training:
            return x
        batch_size, sequence_length, hidden_size = x.size()
        if self.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices((batch_size, sequence_length),
                                                      self.mask_time_prob,
                                                      self.mask_time_length,
                                                      min_masks=2)
            mask_time_indices = torch.tensor(mask_time_indices, device=x.device, dtype=torch.bool)
            flip_mask = torch.rand((batch_size, sequence_length)) > self.observe_time_prob
            x[mask_time_indices & flip_mask.cuda()] = self.masked_spec_embed.to(x.dtype)
        if self.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices((batch_size, hidden_size),
                                                         self.mask_feature_prob,
                                                         self.mask_feature_length,
                                                         min_masks=1)
            mask_feature_indices = torch.tensor(mask_feature_indices, device=x.device, dtype=torch.bool)
            x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        return x


def add_blank(ys_pad: torch.Tensor, blank: int, ignore_id: int) -> torch.Tensor:
    """ Prepad blank for transducer predictor

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        blank (int): index of <blank>

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> blank = 0
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,   4,   5],
                [ 4,  5,  6,  -1,  -1],
                [ 7,  8,  9,  -1,  -1]], dtype=torch.int32)
        >>> ys_in = add_blank(ys_pad, 0, -1)
        >>> ys_in
        tensor([[0,  1,  2,  3,  4,  5],
                [0,  4,  5,  6,  0,  0],
                [0,  7,  8,  9,  0,  0]])
    """
    bs = ys_pad.size(0)
    _blank = torch.tensor([blank], dtype=torch.long, requires_grad=False, device=ys_pad.device)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = torch.cat([_blank, ys_pad], dim=1).long()  # [bs, Lmax+1]
    # print(out.dtype)
    # print(out)
    # print(type(ignore_id))
    return torch.where(out == ignore_id, blank, out)


def get_padding_mask(max_len, batch_size, lengths):
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths[:, None]
    return mask


def get_interval_mask(max_len, start, end, device):
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        start (int): start of interval.
        end (int): end of interval.
    Returns:
        (Tensor): The interval mask with shape [maxlen]
    """
    base = torch.arange(max_len, device=device)
    start_mask = base.ge(start)
    end_mask = base.lt(end)
    return torch.logical_and(start_mask, end_mask)


def unravel_index(
    indices,
    shape,
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices,dim, rounding_mode='trunc')

    coord = torch.stack(coord[::-1], dim=-1)

    return coord