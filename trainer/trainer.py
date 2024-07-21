import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional, Type, Any, Dict

from textbrewer import TrainingConfig, BasicTrainer
from .adaptor import Adaptor
from .validator import Validator


class Trainer:
    """
    Trainer 类封装了模型的训练流程，包括初始化优化器、学习率调度器和训练过程。

    Attributes:
        config (OmegaConf): 配置对象，包含训练相关的配置，如设备、学习率等。
        model (nn.Module): 训练的神经网络模型。
        adaptor (Adaptor): 适配器，用于调整模型输出以适应训练过程。
        device (torch.device): 训练使用的设备。
        validator (Validator): 验证器，用于在训练过程中进行模型验证。
        optimizer (optim.Optimizer): 用于模型的优化器。
        scheduler_class (Type[optim.lr_scheduler._LRScheduler]): 学习率调度器的类。
        scheduler_args (Dict[str, Any]): 传递给学习率调度器的参数。
        train_config (TrainingConfig): TextBrewer的训练配置对象。
        trainer (BasicTrainer): TextBrewer的基础训练器对象。
    """

    def __init__(
        self,
        config: OmegaConf,
        model: nn.Module,
        adaptor: Adaptor,
        scheduler_class: Type[optim.lr_scheduler._LRScheduler],
        scheduler_args: Dict[str, Any],
        validator: Validator,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        """
        初始化Trainer对象。

        Args:
            config (OmegaConf): 包含训练配置的OmegaConf对象。
            model (nn.Module): 要训练的模型。
            adaptor (Adaptor): 模型输出的适配器。
            scheduler_class (Type[optim.lr_scheduler._LRScheduler]): 学习率调度器的类。
            scheduler_args (Dict[str, Any]): 学习率调度器的配置参数。
            validator (Validator): 验证过程的实现。
            optimizer (Optional[optim.Optimizer]): 指定的优化器。如果未提供，则使用默认配置。
        """
        self.config = config
        self.model = model
        self.adaptor = adaptor
        self.device = config.device
        self.validator = validator

        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.learning_rate
            )
        else:
            self.optimizer = optimizer

        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args

        self.train_config = TrainingConfig(
            device=self.device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ckpt_epoch_frequency=config.ckpt_epoch_frequency,
            log_dir=config.log_dir,
            output_dir=config.output_dir,
        )  # 设置训练配置，指定设备

        self.trainer = BasicTrainer(
            train_config=self.train_config,
            model=self.model,
            adaptor=self.adaptor,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        max_grad_norm: float = -1.0,
        batch_postprocessor: Optional[Any] = None,
    ):
        """
        执行模型的训练流程。

        Args:
            train_dataloader (DataLoader): 训练数据的加载器。
            num_epochs (int): 训练的总轮次。
            max_grad_norm (float): 梯度裁剪的最大范数。
            batch_postprocessor (Optional[Any]): 可选的批数据后处理函数。

        使用提供的优化器、学习率调度器和训练配置，结合验证回调进行模型训练。
        """
        with self.trainer:
            self.trainer.train(
                optimizer=self.optimizer,
                dataloader=train_dataloader,
                num_epochs=num_epochs,
                scheduler_class=self.scheduler_class,
                scheduler_args=self.scheduler_args,
                max_grad_norm=max_grad_norm,
                callback=self.validator,
                batch_postprocessor=batch_postprocessor,
            )
