import numpy
import numpy as np
import torch
import random
from torch.utils.data import Dataset

from pathlib import Path
from oct_converter.readers import E2E
from tqdm import tqdm
import pickle
import math
from torchvision import transforms

LABELS = {'HEALTHY': torch.tensor(0), 'SICK': torch.tensor(1)}


tomograms_train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((496, 1024)),  # TODO: Change this to point to config.input_size
    transforms.Normalize(mean=[10.01453], std=[21.36284])
])
tomograms_validation_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((496, 1024)),  # TODO: Change this to point to config.input_size
    transforms.Normalize(mean=[10.53055], std=[21.35372])
])
tomograms_test_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((496, 1024)),  # TODO: Change this to point to config.input_size
    transforms.Normalize(mean=[10.35467], std=[18.61051])  # calculated by util.get_dataset_stats
])

volume_train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((496, 1024)),  # TODO: Change this to point to config.input_size
    transforms.Normalize(mean=[10.01453], std=[21.36284])
])
volume_validation_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((496, 1024)),  # TODO: Change this to point to config.input_size
    transforms.Normalize(mean=[10.53055], std=[21.35372])
])
volume_test_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((496, 1024)),  # TODO: Change this to point to config.input_size
    transforms.Normalize(mean=[10.35467], std=[18.61051])  # calculated by util.get_dataset_stats
])


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

        assert idx < len(self)

        with open(self.cache_fs[idx], 'rb') as file:
            data = pickle.load(file)

        return data, self.labels[idx]

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

    def append(self, value):
        self[len(self)] = value


class PartialCache:
    def __init__(self, cache, lut):
        self.cache = cache
        self.lut = lut

    def get_labels(self):
        return {idx: self.cache.labels[lut_idx] for idx, lut_idx in enumerate(self.lut)}

    def __getitem__(self, idx):
        return self.cache[self.lut[idx]]

    def __setitem__(self, idx, value):
        self.cache[self.lut[idx]] = value

    def __len__(self):
        return len(self.lut)


class DatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.curr_idx = 0

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __next__(self):
        if self.curr_idx == len(self.dataset):
            raise StopIteration

        self.curr_idx += 1
        return self.dataset[self.curr_idx - 1]


class E2EVolumeGenerator(Dataset):
    def __init__(self, cache, transformations=None):
        self.cache = cache
        self.transformations = transformations

    def get_labels(self):
        return self.cache.get_labels()

    def __iter__(self):
        return DatasetIterator(self)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        volume, *other = self.cache[idx]
        if self.transformations is not None:
            volume = self.transformations(volume)

        return volume, *other


class BScansGenerator(Dataset):
    def __init__(self, cache, transformations=None):
        self.cache = cache
        self.transformations = transformations

    def get_labels(self):
        return self.cache.get_labels()

    def __iter__(self):
        return DatasetIterator(self)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        tomogram, *other = self.cache[idx]
        if self.transformations is not None:
            tomogram = self.transformations(tomogram)

        return tomogram.expand(3, -1, -1), *other


def random_split_cache(cache, breakdown):
    """
    :param cache: cache to split
    :param breakdown: list of fractions to breakdown the cache according to
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


def build_volume_cache(cache, path, label):
    """

    :param cache:
    :param path:
    :param label:
    :return:
    """
    counter = 0
    for sample in tqdm(list(Path(path).rglob("*.E2E"))):
        if sample.is_file():
            for scan in E2E(sample).read_oct_volume():
                # Ignore volumes with 0-size
                validity = [isinstance(tomogram, np.ndarray) for tomogram in scan.volume]
                if not all(validity):
                    continue

                try:
                    volume = numpy.moveaxis(scan.volume, 0, 2)  # Works best with Pytorch' transformation
                    cache.append((volume, label))
                    counter += 1
                except Exception as ex:
                    print("Ignored volume {0} in sample {1}. An exception of type {2} occurred. \
                               Arguments:\n{1!r}".format(scan.patient_id, sample, type(ex).__name__, ex.args))
                    continue


def build_tomograms_cache(cache, path, label):
    """
    :param cache:
    :param path:
    :param label:
    :return:
    """

    counter = 0
    for sample in tqdm(list(Path(path).rglob("*.E2E"))):
        if sample.is_file():
            for volume in E2E(sample).read_oct_volume():
                # 'Naively' center around the Fovea
                tomograms_count = len(volume.volume)
                start = max(0, math.ceil(tomograms_count/2) - 1)
                end = min(math.ceil(tomograms_count/2) + 4, tomograms_count)  # python rules: EXCLUDING last one

                # Ignore volumes with 0-size
                validity = [isinstance(volume.volume[idx], np.ndarray) for idx in range(start, end)]
                if not all(validity):
                    continue

                # Save each tomogram to cache
                for idx in range(start, end):
                    try:
                        cache.append((volume.volume[idx], label))
                        counter += 1
                    except Exception as ex:
                        print("Ignored volume {0} in sample {1}. An exception of type {2} occurred. \
                               Arguments:\n{1!r}".format(idx, sample, type(ex).__name__, ex.args))
                        continue


def make_weights_for_balanced_classes(dataset, classes):
    num_classes = len(classes)
    num_scans = len(dataset)
    labels = dataset.get_labels()

    # Count # of appearances per each class
    count = [0] * num_classes
    for label in labels.values():
        count[int(label)] += 1

    # Each class receives weight in reverse proportion to its # of appearances
    weight_per_class = [0.] * num_classes
    for idx in classes.values():
        weight_per_class[idx] = float(num_scans) / float(count[idx])

    # Assign class-corresponding weight for each element
    weights = [0] * num_scans
    for idx, label in labels.items():
        weights[idx] = weight_per_class[int(label)]

    return torch.FloatTensor(weights)


def variable_size_collate(batch):
    data = [volume for volume, _ in batch]
    labels = [label for _, label in batch]
    labels = torch.LongTensor(labels)
    return [data, labels]
