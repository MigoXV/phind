import torch
import h5py
import torch.nn as nn

from omegaconf import OmegaConf
from models import models
from datafeeder.datasets import DatasetNDT, train_collate_fn
from datafeeder.datafeeder import DataFeeder
from datafeeder.batch_postprocessors import TrainBatchPostProcessor

from trainer import Trainer, Validator, WarmupCosineAnnealingLR
from trainer.adaptor import Adaptor
from trainer.reporter import get_reporter
from utils.size_of_model import size_of_model


def train_from_scratch(config):
    # 汇报器
    reporter = get_reporter(
        config.report_to,
        project=config.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )

    df = h5py.File(config.data_dict_json_path, 'r')
    data = zip(df['wave'], df['label'])
    val_data = None
    if config.val_data_dict_json_path:
        df_val = h5py.File(config.val_data_dict_json_path, 'r')
        val_data = zip(df_val['wave'], df_val['label'])

    data_feeder = DataFeeder(
        data_feeder_config=config,
        data=data,
        val_data=val_data,
        reporter=reporter,
        dataset_class=DatasetNDT,
        collate_fn = train_collate_fn
    )
    batch_postprocessor = TrainBatchPostProcessor()
    model = models[config.model_name]().to(config.device)
    # model = Net(hidden_size=256, num_classes=len(ALPHABET) + 1).to(config.device)
    if config.ckpt_path:
        model.load_state_dict(torch.load(config.ckpt_path))
    loss_fn = nn.L1Loss()

    print("The size of model:", size_of_model(model))

    # 适配器
    adaptor = Adaptor(config, loss_fn, reporter=reporter)
    
    # 验证器
    validator = Validator(
        val_dataloader=data_feeder.val_dataloader,
        loss_fn=loss_fn,
        device=config.device,
        reporter=reporter,
        batch_postprocessor=data_feeder.batch_postprocessor,
        adaptor=adaptor
    )

    if config.lr_type == "warmup_cos":
        scheduler_class = WarmupCosineAnnealingLR

        # 学习率调度器
        scheduler_args = {
            "warmup_steps": data_feeder.get_train_step_per_epoch()
            * config.lr_params.warmup_epochs,
            "total_steps": data_feeder.get_max_step(config.num_epochs),
            "min_lr": config.lr_params.min_lr,
            "init_lr": config.learning_rate,
            "reporter": reporter,
        }
    else:
        scheduler_class = None
        scheduler_args = None

    # 训练器
    trainer = Trainer(
        config=config,
        model=model,
        adaptor=adaptor,
        scheduler_class=scheduler_class,
        scheduler_args=scheduler_args,
        validator=validator,
    )

    trainer.train(
        train_dataloader=data_feeder.train_dataloader,
        num_epochs=config.num_epochs,
        max_grad_norm=config.max_grad_norm,
        batch_postprocessor=data_feeder.batch_postprocessor,
    )

    reporter.log_model(
        checkpoint_dir=config.output_dir, name=config.model_name, type="trained-model"
    )
