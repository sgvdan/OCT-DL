import torch
from torch.utils.data import Dataset
from pathlib import Path
from oct_converter.readers import E2E
from tqdm import tqdm

LABELS = {'HEALTHY': 0, 'SICK': 1}


class BScansGenerator(Dataset):
    def __init__(self, control_dir, study_dir):
        self.b_scans = torch.empty(0)
        self.labels = torch.empty(0)
        self.parse_files(control_dir, study_dir)

    def parse_files(self, control_dir, study_dir):
        """
        Appends B-Scans of control and study directories,
        labeled as HEALTHY and SICK correspondingly.
        :param control_dir: path to control directory
        :param study_dir: path to study directory
        :return: None
        """

        print("Load Control")
        for sample in tqdm(list(Path(control_dir).rglob("*.E2E"))):
            if not Path.is_dir(sample):
                for volume in E2E(sample).read_oct_volume():
                    volume_tensor = torch.tensor(volume.volume)
                    self.labels = torch.tensor(LABELS['HEALTHY']).repeat(volume_tensor.shape[0])
                    self.b_scans = torch.cat((self.b_scans, volume_tensor))

        print("Load Study")
        for sample in tqdm(list(Path(study_dir).rglob("*.E2E"))):
            print('SICK:{}'.format(sample))
            if not Path.is_dir(sample):
                for volume in E2E(sample).read_oct_volume():
                    volume_tensor = torch.tensor(volume.volume)
                    self.labels = torch.tensor(LABELS['SICK']).repeat(volume_tensor.shape[0])
                    self.b_scans = torch.cat((self.b_scans, volume_tensor))

    def make_weights_for_balanced_classes(self):
        num_labels = len(LABELS)
        num_scans = len(self)

        count = [0] * num_labels
        for label in self.labels:
            count[label] += 1

        weight_per_label = [0.] * len(LABELS)
        N = float(num_scans)
        for idx in LABELS.values():
            weight_per_label[idx] = N / float(count[idx])

        weights = [0] * num_scans
        for idx, label in enumerate(self.labels):
            weights[idx] = weight_per_label[label]

        return torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.b_scans)

    def __getitem__(self, idx):
        return self.b_scans[idx], self.labels[idx]
