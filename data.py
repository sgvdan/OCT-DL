import torch
from torch.utils.data import Dataset
from pathlib import Path
from oct_converter.readers import E2E

LABELS = {'HEALTHY': 0, 'SICK': 1}


class BScansGenerator(Dataset):
    def __init__(self, control_dir, study_dir):
        self.bscans = torch.empty(0)
        self.parse_files(control_dir, study_dir)

    def parse_files(self, control_dir, study_dir):
        """
        Appends B-Scans of control and study directories,
        labeled as HEALTHY and SICK correspondingly.
        :param control_dir: path to control directory
        :param study_dir: path to study directory
        :return: None
        """
        for sample in list(Path(control_dir).rglob("*.E2E")):
            print('HEALTHY:{}'.format(sample))
            if not Path.is_dir(sample):
                for volume in E2E(sample).read_oct_volume()
                    import pdb; pdb.set_trace()
                    volume_tensor = torch.tensor(volume)
                    self.labels = torch.tensor(LABELS['HEALTHY']).repeat(volume_tensor.shape[0])
                    self.bscans.cat(volume_tensor)
                    # TODO: Transform volume into pytorch tensor
                    # TODO: repeat label to same dimensions as volume
                    # TODO: bscans should be a tuple of (tensor scans, tensor labels) of same length along dim=0
                    for scan in volume.volume:
                        # TODO: Very inefficient. Somehow use extend instead (or Pytorch tensor form)
                        self.bscans.append((scan, LABELS['HEALTHY']))

        for sample in list(Path(study_dir).rglob("*.E2E")):
            print('SICK:{}'.format(sample))
            if not Path.is_dir(sample):
                for volume in E2E(sample).read_oct_volume():
                    # TODO: Transform into pytorch
                    for scan in volume.volume:
                        # TODO: Very inefficient. Somehow use extend instead (or Pytorch tensor form)
                        self.bscans.append((scan, LABELS['SICK']))

    def make_weights_for_balanced_classes(self):
        num_labels = len(LABELS)
        num_scans = len(self)

        count = [0] * num_labels
        for scan, label in self.bscans:
            count[label] += 1

        weight_per_label = [0.] * len(LABELS)
        N = float(num_scans)
        for idx in LABELS.values():
            weight_per_label[idx] = N / float(count[idx])

        weights = [0] * num_scans
        for idx, (_, label) in enumerate(self.bscans):
            weights[idx] = weight_per_label[label]

        return torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.bscans)

    def __getitem__(self, idx):
        return self.bscans[idx]
