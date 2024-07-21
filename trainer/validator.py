import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from .reporter import Reporter
from textbrewer.distiller_utils import move_to_device
from .adaptor import Adaptor



class Validator:


    def __init__(
        self,
        val_dataloader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        batch_postprocessor,
        reporter: Reporter,
        adaptor: Adaptor,
    ) -> None:

        self.dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.reporter = reporter
        self.batch_postprocessor = batch_postprocessor
        self.adaptor = adaptor

    def __call__(self, model: nn.Module, step: int) -> None:
        """
        调用验证过程。

        Args:
            model (nn.Module): 被验证的模型。
            step (int): 当前步骤号。

        Returns:
            None
        """
        metrics = self.validate(model, step)

        self.reporter.report(report_type="val", metrics=metrics)

    def validate(self, model: nn.Module, step) -> dict:
        """
        执行模型验证过程。

        Args:
            model (nn.Module): 被验证的模型。

        Returns:
            dict: 包含验证损失和CER的字典。
        """
        model.eval()
        metrics = {}
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                # 将数据移动到指定设备
                batch = self.batch_postprocessor(batch)
                batch = move_to_device(batch, self.device)
                model_outputs = model(*batch)

                # 计算损失和CER
                for key, value in self.adaptor.get_metrics(
                    batch, model_outputs
                ).items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    if key in metrics:
                        metrics[key] += value / len(self.dataloader)
                    else:
                        metrics[key] = value / len(self.dataloader)

        return metrics
