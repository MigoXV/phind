import os
import math
import torch
import torchaudio

from models.dilation_model import BasicBlock
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch.nn as nn
import torch.nn.functional as F


def get_mask(x: torch.Tensor, x_valid_length: torch.Tensor) -> torch.Tensor:
    """获取Transformer的mask。

    Args:
        x (torch.Tensor): 输入数据，形状为 (batch_size, seq_len, 38)。
        x_valid_length (torch.Tensor): 输入数据的有效长度，形状为 (batch_size, )。

    Returns:
        torch.Tensor: Transformer的mask，形状为 (batch_size, seq_len)。
    """
    seq_len, batch_size, feature_dim = x.shape

    # 初始化mask，全为True
    mask = torch.ones((seq_len, batch_size), dtype=torch.bool, device=x.device)

    # 根据有效长度更新mask
    for i in range(batch_size):
        # valid_len = x_valid_length[i]
        # if valid_len < seq_len:
        mask[x_valid_length[i] :, i] = False  # 将填充位置设置为False

    return mask.T


def create_positional_encoding(max_len, embed_size):
    pos_enc = torch.zeros(max_len, embed_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
    )

    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc.unsqueeze(0)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask
