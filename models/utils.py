import math
import torch

def create_positional_encoding(max_len, embed_size):
    pos_enc = torch.zeros(max_len, embed_size)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
    )

    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc.unsqueeze(0)

