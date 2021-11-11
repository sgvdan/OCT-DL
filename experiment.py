import torch
import random
import data
from config import default_config
from train import train, train_loop
from data import BScansGenerator, Cache
from network import get_model_and_optim

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

        # Set Config TODO: ERASE!
        for key, value in config.items():
            setattr(self, key, value)

        self.config = config

        # Build up data
        self.test_loader, self.validation_loader, self.train_loader = self.buildup_data()

        # Set Model, Optimizer & Loss
        self.criterion = torch.nn.functional.cross_entropy
        self.model, self.optimizer = get_model_and_optim(model_name=self.model_name, lr=self.lr, device=self.device,
                                                         load_best_model=False)

    def buildup_data(self):
        test_cache = Cache('test')
        validation_cache = Cache('validation')
        train_cache = Cache('train')

        if self.refresh_cache:
            data.buildup_cache(test_cache, 'Data/test/control', data.LABELS['HEALTHY'], self.control_limit, self.config)
            data.buildup_cache(test_cache, 'Data/test/study', data.LABELS['SICK'], self.study_limit, self.config)

            data.buildup_cache(validation_cache, 'Data/validation/control', data.LABELS['HEALTHY'], self.control_limit, self.config)
            data.buildup_cache(validation_cache, 'Data/validation/study', data.LABELS['SICK'], self.study_limit, self.config)

            data.buildup_cache(train_cache, 'Data/train/control', data.LABELS['HEALTHY'], self.control_limit, self.config)
            data.buildup_cache(train_cache, 'Data/train/study', data.LABELS['SICK'], self.study_limit, self.config)

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


def main():
    with wandb.init(project="OCT-DL", config=default_config):
        config = wandb.config
        experiment = Experiment(config)
        experiment.run()


if __name__ == '__main__':
    main()
