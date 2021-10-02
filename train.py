from torch.autograd import backward
import wandb
import data
from tqdm import tqdm
from itertools import cycle
import os
import pickle
import torch
from loss import calc_accuracy

MODELS_DIR = os.path.join(".", "Models")
BEST_MODELS_DICT_PATH = os.path.join(MODELS_DIR, "best_models_dict.pkl")


def get_best_models_dict():
    if os.path.exists(BEST_MODELS_DICT_PATH):
        with open(BEST_MODELS_DICT_PATH, "rb") as f:
            best_models_dict = pickle.load(f)
    else:
        best_models_dict = {}
    return best_models_dict


def update_best_models(model, optimizer, model_acc, best_models_dict):
    print("Best model found!!", "Model: {0}, Accuracy: {1}".format(model.name, model_acc))
    # update dict
    best_models_dict[model.name] = model_acc
    with open(BEST_MODELS_DICT_PATH, "wb") as f:
        pickle.dump(best_models_dict, f)
    # update model
    best_model_path = os.path.join(MODELS_DIR, model.name + ".pth")
    print("saving best model to: {0}".format(best_model_path))
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, best_model_path)


def train(model, criterion, optimizer, train_loader, test_loader, epochs, device):
    for epoch in tqdm(range(epochs)):
        for (train_images, train_labels), (test_images, test_labels) in zip(train_loader, cycle(test_loader)):

            # train
            train_model_acc = train_loop(model=model, criterion=criterion,
                                         optimizer=optimizer,
                                         device=device, images=train_images, labels=train_labels,
                                         mode="Train")
            # test
            test_model_acc = train_loop(model=model, criterion=criterion,
                                        optimizer=optimizer,
                                        device=device, images=test_images, labels=test_labels, mode="Test")
            # update best model if necessary
            best_models_dict = get_best_models_dict()
            if model.name not in best_models_dict or test_model_acc > best_models_dict[model.name]:
                update_best_models(model=model, optimizer=optimizer, model_acc=test_model_acc,
                                   best_models_dict=best_models_dict)


def train_loop(model, criterion, optimizer, device, images, labels, mode="Train"):
    # Set model mode
    if mode == "Train":
        model.train()
    elif mode == "Test":
        model.eval()
    else:
        raise RuntimeError()

    # Move to device
    images, labels = images.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.int64)

    # Run the model on the input batch
    pred_scores = model(images)

    # Calculate the accuracy for this batch
    for label, value in data.LABELS.items():
        accuracy = calc_accuracy(pred_scores, labels, specific_label=value)
        wandb.log({"{mode}/accuracy/{label}".format(mode=mode, label=label): accuracy})
    accuracy = calc_accuracy(pred_scores, labels, specific_label=None)
    wandb.log({"{mode}/accuracy".format(mode=mode): accuracy})

    if mode == "Train":
        # Calculate the loss for this batch
        loss = criterion(pred_scores, labels)
        wandb.log({"Train/loss": loss})
        # Update gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return accuracy
