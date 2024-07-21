import typer

app = typer.Typer()

@app.command()
def train(
    dataset: str = typer.Argument(..., help="Path to the dataset hdf5"),
    output_dir: str = typer.Argument(..., help="Path to the model output directory"),
    config_path: str = typer.Argument(..., help="Path to the config file"),
    checkpoint_path: str = typer.Option(
        None, "-c", "--ckpt", help="Path to the checkpoint file"
    ),
    config_set: str = typer.Option(None, "--set", help="Config set"),
    extra_val_dataset: str = typer.Option(None, "-v", "--val", help="Path to the extra val dataset"),
):
    from configs.config_tools import load_train_config

    config = load_train_config(config_path, config_set)
    config.data_dict_json_path = dataset
    config.output_dir = output_dir
    config.ckpt_path = checkpoint_path
    config.val_data_dict_json_path = extra_val_dataset

    from train_ndt import train_from_scratch

    train_from_scratch(config)

if __name__ == "__main__":
    app()