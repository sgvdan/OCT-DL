from torch.utils.data import Dataset
from pathlib import Path
from oct_converter.readers import E2E

LABELS = {'HEALTHY': 0, 'SICK': 1}


class BScansGenerator(Dataset):
    def __init__(self, control_dir, study_dir):
        self.bscans = []
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
                for volume in E2E(sample).read_oct_volume():
                    # TODO: Transform into pytorch tensor
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

    def __len__(self):
        return len(self.bscans)

    def __getitem__(self, idx):
        return self.bscans[idx]
