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

        # Fix randomness
        torch.manual_seed(0)
        random.seed(0)

        # Set Config
        for key, value in config.items():
            setattr(self, key, value)

        # Build up data
        self.test_loader, self.validation_loader, self.train_loader = self.buildup_data()

        # Set Model, Optimizer & Loss
        self.criterion = torch.nn.functional.cross_entropy
        self.model, self.optimizer = get_model_and_optim(model_name=self.model_name, lr=self.lr, device=self.device,
                                                         load_best_model=False)

    def buildup_data(self):
        cache = Cache(self.cache_name)

        if self.refresh_cache:
            transformations = transforms.Resize(self.input_size)
            data.buildup_cache(cache, self.dataset_control_path, data.LABELS['HEALTHY'], self.control_limit, transformations)
            data.buildup_cache(cache, self.dataset_study_path, data.LABELS['SICK'], self.study_limit, transformations)

        test_cache, validation_cache, train_cache = data.random_split_cache(cache, [self.test_size, self.validation_size, self.training_size])

        # Build up Training Dataset
        print("Load Train")
        train_dataset = BScansGenerator(train_cache)
        train_weights = train_dataset.make_weights_for_balanced_classes()
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                        sampler=train_sampler)
        # Build up Validation Dataset
        print("Load Validation")
        validation_dataset = BScansGenerator(validation_cache)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=self.batch_size)

        # Build up Test Dataset
        print("Load Test")
        test_dataset = BScansGenerator(test_cache)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size)

        return test_loader, validation_loader, train_loader

    def run(self):
        self.train_model()
        accuracy = self.eval_model()
        print("Model's accuracy: {}".format(accuracy), flush=True)

    """
    TODO: Doc
    """
    def train_model(self):
        train(model=self.model, criterion=self.criterion,
              optimizer=self.optimizer, train_loader=self.train_loader, validation_loader=self.validation_loader,
              epochs=self.epochs, device=self.device)
        # load_best_state(self.model, self.optimizer)

    def eval_model(self):
        running_test_accuracy = 0.0
        for test_images, test_labels in self.test_loader:
            # test accuracy of batch
            _, accuracy = train_loop(model=self.model, criterion=self.criterion, optimizer=self.optimizer,
                                     device=self.device, images=test_images, labels=test_labels, mode='eval')
            running_test_accuracy += accuracy

        test_accuracy = round(running_test_accuracy/len(self.test_loader), 2)
        wandb.log({"Test/accuracy": test_accuracy})

        return test_accuracy
