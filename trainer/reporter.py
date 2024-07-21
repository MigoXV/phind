import os
import wandb
import torch


class Reporter:
    global_step = 0
    epoch = 0

    def __init__(self, *args, **kwargs) -> None:
        pass

    def report(self, report_type, metrics):
        report_metrics = {}
        if report_type == "lr":
            # report_metrics = {"learing_rate": metrics["lr"]}
            report_metrics["learing_rate"] = metrics["lr"]

        else:
            if report_type == "train":
                Reporter.global_step += 1
            if report_type == "val":
                Reporter.epoch += 1
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    report_metrics[report_type + "_" + key] = value.item()
                else:
                    report_metrics[report_type + "_" + key] = value

        report_metrics["epoch"] = Reporter.epoch
        report_metrics["global_step"] = Reporter.global_step

        print(report_metrics)

        return report_metrics

    def log_artifact(self, name, data, **kwargs):
        print("log artifact")

    def log_dataset(self, dataset, **kwargs):
        print("log dataset")

    def log_model(self, checkpoint_dir, **kwargs):
        print("log model")


class WandbReporter(Reporter):
    def __init__(
        self, project: str, config: dict = None, entity: str = None, tags: list = None
    ) -> None:
        self.run = wandb.init(project=project, config=config, entity=entity, tags=tags)
        super().__init__()

    def report(self, report_type, metrics):
        report_metrics = super().report(report_type, metrics)
        self.run.log(report_metrics, step=Reporter.global_step)

    def log_dataset(self, dataset_path, **kwargs):
        dataset_artifact = wandb.Artifact(
            name=kwargs.get("name"),
            type=kwargs.get("type"),
            description=kwargs.get("description"),
        )

        dataset_artifact.add_file(dataset_path)

        self.run.log_artifact(dataset_artifact)

        super().log_dataset(dataset_path, **kwargs)

    def log_model(self, checkpoint_dir, **kwargs):
        model_artifact = wandb.Artifact(
            name=kwargs.get("name"),
            type=kwargs.get("type"),
            description=kwargs.get("description"),
        )

        # 将路径转换为绝对路径
        checkpoint_dir = os.path.abspath(checkpoint_dir)

        model_artifact.add_reference(uri="file://" + checkpoint_dir)

        self.run.log_artifact(model_artifact)

        super().log_model(checkpoint_dir, **kwargs)


def get_reporter(report_to: str, *args, **kargs) -> Reporter:
    if report_to == "wandb":
        return WandbReporter(*args, **kargs)
    else:
        return Reporter(*args, **kargs)
