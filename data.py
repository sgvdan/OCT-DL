import torch
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path
from oct_converter.readers import E2E
from tqdm import tqdm
import pickle
import math
import util
import wandb

LABELS = {'HEALTHY': 0, 'SICK': 1}


class Cache:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.cache_fs_path = self.cache_path / '.cache_fs'
        self.labels_path = self.cache_path / '.labels'

        # Create cache directory
        if not self.cache_path.exists():
            self.cache_path.mkdir()

        # Retrieve file-system dictionary
        if self.cache_fs_path.exists():
            with open(self.cache_fs_path, 'rb') as file:
                self.cache_fs = pickle.load(file)
        else:
            self.cache_fs = {}

        # Retrieve labels
        if self.labels_path.exists():
            with open(self.labels_path, 'rb') as file:
                self.labels = pickle.load(file)
        else:
            self.labels = {}

    def buildup_data(self, path, label, transformations):
        counter = len(self)
        for sample in tqdm(list(Path(path).rglob("*.E2E"))):
            if not Path.is_dir(sample):
                for volume in E2E(sample).read_oct_volume():

                    # Center around the Fovea
                    tomograms_count = len(volume.volume)
                    start = max(0, math.ceil(tomograms_count/2) - 1)
                    end = min(math.ceil(tomograms_count/2) + 4, tomograms_count)  # python rules: EXCLUDING last one

                    # Save to cache
                    for idx in range(start, end):
                        try:
                            # TODO: Incorporate all transformations to 'transformations'
                            tomogram = transformations(torch.tensor(volume.volume[idx]).unsqueeze(0).expand(3, -1, -1))
                            self[counter] = tomogram
                            self.labels[counter] = label
                            counter += 1
                        except:
                            print("Ignored volume {0} in sample {1} - type match error".format(idx, sample))
                            continue

                    # util.imshow(volume.volume[start], "{0} START - {1}:{2}".format(label, sample, volume.patient_id))
                    # util.imshow(volume.volume[end - 1], "{0} END - {1}:{2}".format(label, sample, volume.patient_id))

                    wandb.log({'label{0}-start'.format(label): [wandb.Image(volume.volume[start])],
                               'label{0}-end'.format(label): [wandb.Image(volume.volume[end - 1])]})

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        with open(self.cache_fs[idx], 'rb') as file:
            return pickle.load(file)

    def __setitem__(self, idx, value):
        item_path = self.cache_path / str(idx)
        with open(item_path, 'wb+') as file:
            pickle.dump(value, file)

        self.cache_fs[idx] = item_path

    def __len__(self):
        return len(self.cache_fs)

    def __del__(self):
        with open(self.cache_fs_path, 'wb+') as file:
            pickle.dump(self.cache_fs, file)

        with open(self.labels_path, 'wb+') as file:
            pickle.dump(self.labels, file)


class BScansGenerator(Dataset):
    def __init__(self, control_dir, study_dir, input_size):
        self.b_scans = torch.empty(0)
        self.labels = torch.empty(0)

        self.cache = Cache('./cache')
        print("Buildup Control Cache", flush=True)
        self.cache.buildup_data(control_dir, torch.tensor(LABELS['HEALTHY']), transforms.Resize(input_size))
        print("Buildup Study Cache", flush=True)
        self.cache.buildup_data(study_dir, torch.tensor(LABELS['SICK']), transforms.Resize(input_size))

        self.labels = self.cache.get_labels()

    def make_weights_for_balanced_classes(self):
        num_labels = len(LABELS)
        num_scans = len(self.cache)

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
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx], self.labels[idx]
