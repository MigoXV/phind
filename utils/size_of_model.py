def size_of_model(model):
    num_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if num_of_parameters < 1e6:
        num_of_parameters = f"{int(num_of_parameters / 1e3)}K"
    elif num_of_parameters < 1e9:
        num_of_parameters = f"{int(num_of_parameters / 1e6)}M"
    elif num_of_parameters < 1e12:
        num_of_parameters = f"{int(num_of_parameters / 1e9)}B"

    return num_of_parameters