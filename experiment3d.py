import torch
import random
import data
from config import default_config
from set_network import E2ESetNetwork
from train import evaluate, train
from data import E2EVolumeGenerator, Cache, make_weights_for_balanced_classes
from network import get_model_and_optim, load_best_state

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

        self.config = config

        # Build up data
        self.test_loader, self.validation_loader, self.train_loader = self.buildup_data()

        # Set Model, Optimizer & Loss
        self.criterion = torch.nn.functional.cross_entropy
        self.backbone, _ = get_model_and_optim(model_name=self.config.model_name, lr=self.config.lr,
                                               device=self.config.device, load_best_model=True, pretrained=False)
        self.model = E2ESetNetwork(self.backbone, 0, 0, len(data.LABELS))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def buildup_data(self):
        test_cache = Cache('volume-test')
        validation_cache = Cache('volume-validation')
        train_cache = Cache('volume-train')

        if self.config.refresh_cache:
            data.build_volume_cache(test_cache, 'Data/test/control', data.LABELS['HEALTHY'])
            data.build_volume_cache(test_cache, 'Data/test/study', data.LABELS['SICK'])

            data.build_volume_cache(validation_cache, 'Data/validation/control', data.LABELS['HEALTHY'])
            data.build_volume_cache(validation_cache, 'Data/validation/study', data.LABELS['SICK'])

            data.build_volume_cache(train_cache, 'Data/train/control', data.LABELS['HEALTHY'])
            data.build_volume_cache(train_cache, 'Data/train/study', data.LABELS['SICK'])

        # Build up Training Dataset
        print("Load Train")
        train_dataset = E2EVolumeGenerator(train_cache, transformations=data.volume_train_transformation)
        train_weights = make_weights_for_balanced_classes(train_dataset, data.LABELS)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.config.batch_size,
                                                   sampler=train_sampler)
        # Build up Validation Dataset
        print("Load Validation")
        validation_dataset = E2EVolumeGenerator(validation_cache, transformations=data.volume_validation_transformation)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=self.config.batch_size)

        # Build up Test Dataset
        print("Load Test")
        test_dataset = E2EVolumeGenerator(test_cache, transformations=data.volume_test_transformation)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.config.batch_size)

        return test_loader, validation_loader, train_loader

    def run(self):
        self.train_model()
        load_best_state(self.model, self.optimizer)  # Model Evaluation is performed on best-validation model
        accuracy = self.eval_model()
        print("Model's accuracy: {}".format(accuracy), flush=True)

    """
    TODO: Doc
    """
    def train_model(self):
        train(model=self.model, criterion=self.criterion,
              optimizer=self.optimizer, train_loader=self.train_loader, validation_loader=self.validation_loader,
              epochs=self.config.epochs, device=self.config.device)

    def eval_model(self):
        return evaluate(self.model, self.test_loader, 'Test', device=self.config.device)


def runner(config):
    experiment = Experiment(config)
    experiment.run()


def main():
    # with wandb.init(project="OCT-DL", config=default_config):
    #     runner(wandb.config)
    runner(default_config)


if __name__ == '__main__':
    main()
