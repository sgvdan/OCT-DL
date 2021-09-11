from torch.utils.data import Dataset
from pathlib import Path
from oct_converter.readers import E2E


class OCTGenerator(Dataset):
    def __init__(self, dir):
        self.files = list(Path(".").rglob("*.E2E"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return E2E(self.files[idx])

    def hello(self):
        print(self.files)
