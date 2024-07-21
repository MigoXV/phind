import typer
from scipy.io import loadmat

app = typer.Typer()

@app.command()
def read_mat_file(file_path: str = typer.Argument(..., help="Path to the .mat file")):
    """
    读取 .mat 文件并输出其中所有变量的名称和形状。
    """
    mat_contents = loadmat(file_path)
    
    # 过滤掉特殊的全局变量（以 '__' 开头和结尾）
    variables = {k: v for k, v in mat_contents.items() if not k.startswith('__')}

    if not variables:
        typer.echo("No variables found in the .mat file.")
        return

    typer.echo(f"Variables in {file_path}:")
    for name, data in variables.items():
        typer.echo(f"{name}: shape {data.shape}")



if __name__ == "__main__":
    app()
