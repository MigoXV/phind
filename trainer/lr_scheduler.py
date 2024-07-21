import torch
import torch.nn as nn
import torch.optim as optim
import math

from .reporter import Reporter
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    自定义学习率调度器，结合了线性预热和余弦退火策略。

    详细描述:
        该调度器首先在预热阶段线性增加学习率，然后在余弦退火阶段逐渐减少学习率。

    属性:
        warmup_steps (int): 预热的step数。
        total_steps (int): 总的step数。
        min_lr (float): 最低学习率。

    方法:
        get_lr(): 计算并返回当前step的学习率。

    示例:
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps=500, total_steps=10000)
        >>> for step in range(10000):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    注意:
        该调度器需要在每个step结束后调用`step()`方法来更新学习率。
    """

    def __init__(
        self,
        init_lr: float,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
        last_step: int = -1,
        reporter: Reporter = Reporter(),
    ):
        """
        初始化WarmupCosineAnnealingLR。

        参数:
            optimizer (optim.Optimizer): 优化器实例。
            warmup_steps (int): 预热的step数。
            total_steps (int): 总的step数。
            min_lr (float, optional): 最低学习率。默认值为0。
            last_epoch (int, optional): 上一个epoch的索引。默认值为-1。
        """
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.reporter = reporter
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_step)

    def get_lr(self) -> list:
        """
        计算并返回当前step的学习率。

        返回:
            list: 当前step的学习率列表。
        """
        if self.last_epoch < self.warmup_steps:
            # 线性预热阶段的学习率计算
            self.reporter.report(
                report_type="lr",
                metrics={
                    "lr": self.init_lr * (self.last_epoch + 1) / self.warmup_steps
                },
            )
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # 余弦退火阶段的学习率计算
            cos_anneal_step = self.last_epoch - self.warmup_steps
            cos_total_steps = self.total_steps - self.warmup_steps

            self.reporter.report(
                report_type="lr",
                metrics={
                    "lr": self.min_lr
                    + (self.init_lr - self.min_lr)
                    * (1 + math.cos(math.pi * cos_anneal_step / cos_total_steps))
                    / 2
                },
            )
            return [
                self.min_lr
                + (base_lr - self.min_lr)
                * (1 + math.cos(math.pi * cos_anneal_step / cos_total_steps))
                / 2
                for base_lr in self.base_lrs
            ]


def get_lr_scheduler(scheduler_type: str):
    if scheduler_type == "warmup_cos":
        return WarmupCosineAnnealingLR
    else:
        return None
