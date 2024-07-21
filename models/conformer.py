import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_positional_encoding

class Net(nn.Module):
    def __init__(self, hidden_size = 256, drop_prob: float = 0.1):
        """初始化Net模型。

        Args:
            hidden_size (int): Transformer隐藏层的大小。
            num_classes (int): 输出类别的数量。
            drop_prob (float): Dropout概率。
        """
        super(Net, self).__init__()
        self.hidden_size = hidden_size

        # 输入层
        self.fc1 = nn.Linear(41, self.hidden_size)

        self.conformer = torchaudio.models.Conformer(
            input_dim=hidden_size,
            num_heads=16,
            num_layers=4,
            dropout=drop_prob,
            depthwise_conv_kernel_size=33,
            ffn_dim=512,
        )
        # 输出层
        self.fc2 = nn.Linear(hidden_size, 1)
        self.positional_encoding = create_positional_encoding(
            5000, hidden_size
        )        

    def forward(self, *batch: torch.Tensor) -> torch.Tensor:
        """模型的前向传播过程。

        Args:
            *batch (torch.Tensor): 输入的批数据。

        Returns:
            torch.Tensor: 前向传播的输出结果。
        """
        x = batch[0]
        
        input_length = batch[1]
        x = self.fc1(x)
        x = x + self.positional_encoding[:, : x.size(1), :].to(x.device)
        x, _ = self.conformer(x, input_length)
        x = self.fc2(x[:,0:1,:])
        x = x.squeeze()
        return x
