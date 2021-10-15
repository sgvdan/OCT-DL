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


def train(model, criterion, optimizer, train_loader, validation_loader, epochs, device):
    for epoch in tqdm(range(epochs)):
        running_train_loss, running_train_accuracy = 0.0, 0.0
        for train_images, train_labels in train_loader:
            # train
            loss, accuracy = train_loop(model=model, criterion=criterion, optimizer=optimizer, device=device,
                                        images=train_images, labels=train_labels, mode='train')

            running_train_loss += loss.item()
            running_train_accuracy += accuracy
        wandb.log({'Train/loss': running_train_loss/len(train_loader),
                   'Train/accuracy': running_train_accuracy/len(train_loader)})

        running_validation_accuracy = 0.0
        for validation_images, validation_labels in validation_loader:
            _, accuracy = train_loop(model=model, criterion=criterion, optimizer=optimizer, device=device,
                                     images=validation_images, labels=validation_labels, mode='eval')
            running_validation_accuracy += accuracy
        wandb.log({'Validation/accuracy': running_validation_accuracy/len(train_loader)})


def train_loop(model, criterion, optimizer, device, images, labels, mode):
    # Set model mode
    if mode == "train" and not model.training:
        model.train()
    elif mode == "eval" and model.training:
        model.eval()

    # Move to device
    images, labels = images.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.int64)

    # Run the model on the input batch
    pred_scores = model(images)

    # Calculate the loss & accuracy
    loss = criterion(pred_scores, labels)
    accuracy = calc_accuracy(pred_scores, labels, specific_label=None)

    if mode == "train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, accuracy
