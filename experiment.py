import torch
import random
import data
from train import train, train_loop
from data import BScansGenerator, Cache
from network import get_model_and_optim
from torchvision import transforms

import wandb

wandb.login()


class Experiment:
    """
    TODO: DOC.
    """

    def __init__(self, config):
        assert config is not None

        torch.manual_seed(0)
        random.seed(0)

        for key, value in config.items():
            setattr(self, key, value)

        train_cache, test_cache = self.buildup_data()

        # Train Dataset
        print("Load train dataset")
        self.train_dataset = BScansGenerator(train_cache)
        train_weights = self.train_dataset.make_weights_for_balanced_classes()
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        sampler=train_sampler)

        # TODO: Split to validation and training. Code from Michal:
        # torch.manual_seed(0)
        # self.train_set, self.val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
        #                                                                        len(dataset) - int(0.8 * len(dataset))])

        # Test Dataset
        print("Load test dataset")
        self.test_dataset = BScansGenerator(test_cache)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                                       shuffle=True)

        # Set Model, Optimizer & Loss
        self.criterion = torch.nn.functional.cross_entropy
        self.model, self.optimizer = get_model_and_optim(model_name=self.model_name, lr=self.lr, device=self.device,
                                                         load_best_model=False)

    def run(self):
        self.train_model()
        self.model_acc = self.eval_model()
        wandb.log({"model_accuracy": self.model_acc})
        print("Model's accuracy: {}".format(self.model_acc), flush=True)

    """
    TODO: Doc
    """
    def train_model(self):
        train(model=self.model, criterion=self.criterion,
              optimizer=self.optimizer, train_loader=self.train_loader, test_loader=self.test_loader,
              epochs=self.epochs, device=self.device)
        # load_best_state(self.model, self.optimizer)

    """
    TODO: Doc
    """
    def eval_model(self):
        accuracies = []
        weights = []
        for (test_images, test_labels) in self.test_loader:
            # test accuracy of batch
            accuracy = train_loop(model=self.model, criterion=self.criterion,
                                  optimizer=self.optimizer, device=self.device,
                                  images=test_images, labels=test_labels, mode="Test")
            accuracies.append(accuracy)
            weights.append(test_images.size(0))
        test_accuracy = sum([accuracy * weight for accuracy, weight in zip(accuracies, weights)]) / sum(weights)
        return round(test_accuracy, 2)

    def buildup_data(self):
        """
        :return: (test_cache, validation_cache, train_cache)
        """
        cache = Cache(self.cache_path, 'train')
        # test_cache = Cache(self.cache_path, 'test')

        if self.refresh_cache:
            transformations = transforms.Resize(self.input_size)
            data.buildup_cache(cache, self.train_path['control'], data.LABELS['HEALTHY'], transformations)
            data.buildup_cache(cache, self.train_path['study'], data.LABELS['SICK'], transformations)

        return data.random_split_cache(cache, [0.2, 0.15, 0.65])
