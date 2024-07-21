from torch.utils.data import Dataset

def predict_collate_fn(batch: list) -> tuple:
    signals = zip(*batch)

    return signals

def train_collate_fn(batch: list) -> tuple:
    signals, labels = zip(*batch)

    return signals, labels

class DatasetNDT(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]