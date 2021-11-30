import numpy as np
import wandb
import data
from tqdm import tqdm
import numpy
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
    # update dict
    best_models_dict[model.name] = model_acc
    with open(BEST_MODELS_DICT_PATH, "wb+") as f:
        pickle.dump(best_models_dict, f)
    # update model
    best_model_path = os.path.join(MODELS_DIR, model.name + ".pth")
    print("saving best model to: {0}".format(best_model_path))
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, best_model_path)


def train(model, criterion, optimizer, train_loader, validation_loader, epochs, device):
    for epoch in tqdm(range(epochs)):
        # Train
        running_train_loss = numpy.zeros(1)
        running_train_accuracy = numpy.zeros(len(data.LABELS))

        # TODO: SORT OUT THE CODE SO IT WILL BE POSSIBLE TO GO BY WITH A LIST?

        for train_images, train_labels in train_loader:
            loss, accuracy = train_loop(model=model, criterion=criterion, optimizer=optimizer, device=device,
                                        images=train_images, labels=train_labels, mode='train')

            running_train_loss += loss
            running_train_accuracy += accuracy
        wandb.log({'Train/loss': running_train_loss/len(train_loader),
                   'Train/epoch': epoch})

        for idx, label_key in enumerate(data.LABELS.keys()):
            wandb.log({'Train/accuracy/{}'.format(label_key): running_train_accuracy[idx]/len(train_loader)})

        # Validate
        avg_accuracy = np.average(evaluate(model, validation_loader, 'Validation', device=device))

        # Update best models if needed
        best_models_dict = get_best_models_dict()
        if model.name not in best_models_dict or avg_accuracy > best_models_dict[model.name]:
            print("Best model found!", "Model: {0}, Avg. Accuracy: {1}".format(model.name, avg_accuracy))
            update_best_models(model=model, optimizer=optimizer, model_acc=avg_accuracy,
                               best_models_dict=best_models_dict)


def evaluate(model, dataset_loader, title, device):
    running_validation_accuracy = numpy.zeros(len(data.LABELS))
    for images, labels in dataset_loader:
        _, accuracy = train_loop(model=model, criterion=None, optimizer=None, device=device,
                                 images=images, labels=labels, mode='eval')
        running_validation_accuracy += accuracy

    total_accuracy = running_validation_accuracy / len(dataset_loader)
    for idx, label_key in enumerate(data.LABELS.keys()):
        wandb.log({'{title}/Accuracy/{label_key}'.format(title=title, label_key=label_key): total_accuracy[idx]})

    return total_accuracy


def train_loop(model, criterion, optimizer, device, images, labels, mode):
    # Make sure mode is as expected
    if mode == "train" and not model.training:
        model.train()
    elif mode == "eval" and model.training:
        model.eval()

    # Move to device
    images, labels = images.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.int64)

    # Run the model on the input batch
    pred_scores = model(images)

    # Calculate the loss & accuracy
    accuracy = numpy.array([calc_accuracy(pred_scores, labels, specific_label=label) for label in data.LABELS.values()])
    loss_value = 0

    if mode == "train":
        loss = criterion(pred_scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

    return loss_value, accuracy
