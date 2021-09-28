import torch
from train import train, train_loop
from data import BScansGenerator
from network import get_model_and_optim, load_best_state
import wandb

wandb.login()


class Experiment:
    """
    TODO: DOC.
    """

    def __init__(self, config):
        assert config is not None

        for key, value in config.items():
            setattr(self, key, value)

        # train
        self.train_dataset = BScansGenerator(control_dir=self.train_path['control'], study_dir=self.train_path['study'],
                                             input_size=self.input_size)
        train_weights = self.train_dataset.make_weights_for_balanced_classes()
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size,
                                                        sampler=train_sampler)
        # test
        self.test_dataset = BScansGenerator(control_dir=self.test_path['control'], study_dir=self.test_path['study'],
                                            input_size=self.input_size)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size,
                                                       shuffle=True)

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
