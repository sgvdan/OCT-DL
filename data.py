import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path
from oct_converter.readers import E2E
from tqdm import tqdm
import numpy as np
import math
import util
import wandb

LABELS = {'HEALTHY': 0, 'SICK': 1}


class BScansGenerator(Dataset):
    def __init__(self, control_dir, study_dir, input_size):
        self.b_scans = torch.empty(0)
        self.labels = torch.empty(0)

        print("Load Control", flush=True)
        self.parse_files(control_dir, LABELS['HEALTHY'], input_size)
        print("Load Study", flush=True)
        self.parse_files(study_dir, LABELS['SICK'], input_size)

    def parse_files(self, path, label, input_size):
        """
        Appends B-Scans of a given directory recursively, labeled with the given label
        :param path: path to data directory
        :param label: the label to assign to all parsed B-Scans
        :param input_size: the size to which the data should be fitted

        :return: None
        """
        resize_volume = transforms.Resize(input_size)

        for sample in tqdm(list(Path(path).rglob("*.E2E"))):
            if not Path.is_dir(sample):
                for volume in E2E(sample).read_oct_volume():
                    # TODO: change this to pytorch transformations

                    # Center around the Fovea
                    tomograms_count = len(volume.volume)
                    start = max(0, math.ceil(tomograms_count/2) - 1)
                    end = min(math.ceil(tomograms_count/2) + 4, tomograms_count)  # python rules: EXCLUDING last one
                    try:
                        volume_tensor = resize_volume(torch.tensor(volume.volume[start:end])).unsqueeze(1).expand(-1, 3, -1, -1) # Copying it to hold the same value throught all RGB dimensions
                    except:
                        print("Ignored volume in sample {0} since its type doesn't match tensors".format(sample))
                        continue

                    labels_tensor = torch.tensor(label).repeat(volume_tensor.shape[0])
                    self.labels = torch.cat((self.labels, labels_tensor))
                    self.b_scans = torch.cat((self.b_scans, volume_tensor))
                    # util.imshow(volume.volume[start], "{0} START - {1}:{2}".format(label, sample, volume.patient_id))
                    # util.imshow(volume.volume[end - 1], "{0} END - {1}:{2}".format(label, sample, volume.patient_id))

                    wandb.log({'label{0}-start'.format(label): [wandb.Image(volume.volume[start])],
                               'label{0}-end'.format(label): [wandb.Image(volume.volume[end - 1])]})

    def make_weights_for_balanced_classes(self):
        # TODO: Change this to go through directories and decide weights by that
        num_labels = len(LABELS)
        num_scans = len(self)

        count = [0] * num_labels
        for label in self.labels:
            count[int(label)] += 1

        weight_per_label = [0.] * len(LABELS)
        N = float(num_scans)
        for idx in LABELS.values():
            weight_per_label[idx] = N / float(count[idx])

        weights = [0] * num_scans
        for idx, label in enumerate(self.labels):
            weights[idx] = weight_per_label[int(label)]

        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.b_scans)

    def __getitem__(self, idx):
        return self.b_scans[idx], self.labels[idx]
