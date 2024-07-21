import torch
from torch import nn
from omegaconf import OmegaConf
from .reporter import Reporter

class Adaptor:

    def __init__(
        self, config: OmegaConf, loss_fn: nn.Module, reporter: Reporter = Reporter()
    ) -> None:

        self.loss_fn = loss_fn
        self.reporter = reporter

    def __call__(self, batch: tuple, model_outputs: tuple) -> dict:

        metrics = self.get_metrics(batch, model_outputs)

        self.reporter.report(metrics=metrics, report_type="train")  # 记录损失和CER

        return metrics

    def get_metrics(self, batch: tuple, model_outputs: torch.Tensor) -> dict:
        # 解包batch和model_outputs
        _, *_, labels = batch

        loss = self.loss_fn(model_outputs, labels)

        metrics = {
            "losses": loss,
        }

        return metrics
