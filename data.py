import torch
import random
from torch.utils.data import Dataset
import math

from pathlib import Path
from oct_converter.readers import E2E
from tqdm import tqdm
import pickle
import math
import util
import wandb

LABELS = {'HEALTHY': torch.tensor(0), 'SICK': torch.tensor(1)}


class Cache:
    def __init__(self, name):
        """
        :param name: Unique name for this cache
        """
        self.cache_path = Path('./.cache') / name
        self.cache_fs_path = self.cache_path / '.cache_fs'
        self.labels_path = self.cache_path / '.labels'

        # Create cache directory
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)

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

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        """
        :param idx: idx of data
        :return: returns the data stored in cache's idx
        """
        with open(self.cache_fs[idx], 'rb') as file:
            data = pickle.load(file)

        return data

    def __setitem__(self, idx, value):
        """
        :param idx: index to set by
        :param value: (data, label) - tuple of data and label, both should be torch tensors
        :return: None
        """
        data, label = value

        item_path = self.cache_path / str(idx)
        with open(item_path, 'wb+') as file:
            pickle.dump(data, file)

        self.cache_fs[idx] = item_path
        self.labels[idx] = label

        with open(self.cache_fs_path, 'wb+') as file:
            pickle.dump(self.cache_fs, file)

        with open(self.labels_path, 'wb+') as file:
            pickle.dump(self.labels, file)

    def __len__(self):
        return len(self.cache_fs)


class PartialCache:
    def __init__(self, cache, lut):
        self.cache = cache
        self.lut = lut
        self.labels = {idx: self.cache.labels[lut_idx] for idx, lut_idx in enumerate(lut)}

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        return self.cache[self.lut[idx]]

    def __setitem__(self, idx, value):
        self.cache[self.lut[idx]] = value

    def __len__(self):
        return len(self.lut)


class BScansGenerator(Dataset):
    def __init__(self, cache):
        self.cache = cache
        self.labels = cache.get_labels()

    def make_weights_for_balanced_classes(self):
        num_labels = len(LABELS)
        num_scans = len(self.cache)

        count = [0] * num_labels
        for label in self.labels.values():
            count[int(label)] += 1

        weight_per_label = [0.] * len(LABELS)
        N = float(num_scans)
        for idx in LABELS.values():
            weight_per_label[idx] = N / float(count[idx])

        weights = [0] * num_scans
        for idx, label in self.labels.items():
            weights[idx] = weight_per_label[int(label)]

        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx], self.labels[idx]


def random_split_cache(cache, breakdown):
    """

    :param cache: cache to split
    :param breakdown: list of fractions to breakdown the cache according tobreakdowns
    :return: list of caches, generated from input *cache* according to *breakdown*
    """
    assert sum(breakdown) == 1

    # Randomly split the caches
    lut = list(range(len(cache)))
    random.shuffle(lut)

    caches = []
    tail = 0
    for part in breakdown:
        head = math.floor(tail + part * len(cache))  # we might be losing `len(breakdown)` samples here - negligible
        caches.append(PartialCache(cache, lut[tail:head]))
        tail = head

    return caches


def buildup_cache(cache, path, label, limit, transformations):
    """

    :param cache:
    :param path:
    :param label:
    :param transformations:
    :return:
    """
    counter = len(cache)
    for sample in tqdm(list(Path(path).rglob("*.E2E"))):
        if not Path.is_dir(sample):
            for volume in E2E(sample).read_oct_volume():
                # 'Naively' center around the Fovea
                tomograms_count = len(volume.volume)
                start = max(0, math.ceil(tomograms_count/2) - 1)
                end = min(math.ceil(tomograms_count/2) + 4, tomograms_count)  # python rules: EXCLUDING last one

                # Save each tomogram to cache
                for idx in range(start, end):
                    try:
                        # TODO: Incorporate all transformations to 'transformations'
                        tomogram = transformations(torch.tensor(volume.volume[idx]).unsqueeze(0).expand(3, -1, -1))
                        cache[counter] = (tomogram, label)
                        counter += 1
                    except Exception as ex:
                        print("Ignored volume {0} in sample {1}. An exception of type {2} occurred. \
                               Arguments:\n{1!r}".format(idx, sample, type(ex).__name__, ex.args))
                        continue

                    if counter >= limit:
                        return

                # util.imshow(volume.volume[start], "{0} START - {1}:{2}".format(label, sample, volume.patient_id))
                # util.imshow(volume.volume[end - 1], "{0} END - {1}:{2}".format(label, sample, volume.patient_id))
                # OR
                # wandb.log({'label{0}-start'.format(label): [wandb.Image(volume.volume[start])],
                #            'label{0}-end'.format(label): [wandb.Image(volume.volume[end - 1])]})
