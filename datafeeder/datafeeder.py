from omegaconf import OmegaConf
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import List, Dict, Union, Callable
from sklearn.model_selection import train_test_split


@dataclass
class DataFeederConfig:
    """
    数据馈送器配置类

    Attributes:
        test_size (float): 测试数据集所占比例。
        train_batch_size (int): 训练集的批次大小。
        val_batch_size (int): 验证集的批次大小。
        num_workers (int): 数据加载时使用的工作线程数。
    """

    test_size: float
    train_batch_size: int
    val_batch_size: int
    num_workers: int


class DataFeeder:
    """
    DataFeeder类用于根据给定的配置和数据生成训练和验证的数据加载器。

    Attributes:
        train_dataloader (DataLoader): 用于训练的数据加载器。
        val_dataloader (DataLoader): 用于验证的数据加载器。

    Methods:
        get_train_step_per_epoch(): 获取每个训练轮次的步骤数。
        get_max_step(epoch: int): 获取最大步骤数。
    """

    def __init__(
        self,
        datafeeder_config: Union[DataFeederConfig, OmegaConf],
        data: List[Dict],
        dataset_class,
        collate_fn: Callable,
        val_data: List[Dict] = [],
    ):


        # 若传入val_data，则使用指定的测试集，否则根据test_size划分数据集
        if val_data:
            # extra_val = json.load(open(val_data, "r"))
            train_dataset = dataset_class(data)
            val_dataset = dataset_class(val_data)
        else:
            # 若test_size为0，则全为训练集；若test_size为1，则全为测试集；否则按照test_size划分数据集
            if datafeeder_config.test_size == 0:
                train_dataset = dataset_class(data)
                val_dataset = None
            elif datafeeder_config.test_size == 1:
                train_dataset = None
                val_dataset = dataset_class(data)
            else:
                train_data, val_data = train_test_split(
                    data, test_size=datafeeder_config.test_size
                )
                train_dataset = dataset_class(train_data)
                val_dataset = dataset_class(val_data)

        # 创建数据加载器
        if train_dataset:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=datafeeder_config.train_batch_size,
                shuffle=True,
                num_workers=datafeeder_config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                # drop_last=True,
            )

        if val_dataset is not None:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=datafeeder_config.val_batch_size,
                shuffle=False,
                num_workers=datafeeder_config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                # drop_last=True,
            )

    def get_train_step_per_epoch(self) -> int:
        """
        获取每个训练轮次的步骤数。

        Args:
            None

        Returns:
            int: 每个训练轮次的步骤数。
        """
        return len(self.train_dataloader)

    def get_max_step(self, epoch: int) -> int:
        """
        获取最大步骤数。

        Args:
            epoch (int): 训练轮次。

        Returns:
            int: 最大步骤数。
        """
        return len(self.train_dataloader) * epoch