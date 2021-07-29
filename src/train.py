import copy
import time
import torch
import torchvision
import numpy as np

import utils
from config import config
from test import test


def train(dataloaders, model, optimizer, criterion, exp_name, wandb=None):

    num_epochs = config["num_epochs"]

    # all_results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # best_train_loss, best_train_acc = 0, 0
    # best_val_loss, best_val_acc = 0, 0


    for epoch in range(num_epochs):
        utils.log("\nEpoch {}/{}:".format(epoch+1, num_epochs))

        for phase in ["train", "val"]:
            model.train(phase == "train")

            epoch_loss, epoch_acc = 0, 0
            batch_acc, batch_loss = 0, 0
            
            for inputs, targets in dataloaders[phase]:
                if torch.cuda.is_available(): 
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    outputs = torch.sigmoid(outputs)
                    targets = targets.unsqueeze(1).float()

                    loss = criterion(outputs, targets)
                    predictions = torch.round(outputs)
                    #_, predictions = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_loss += loss.item() * len(inputs)
                batch_acc += torch.mean((predictions == targets) * 1.0) * len(inputs)

            epoch_loss = batch_loss / len(dataloaders[phase].dataset)
            epoch_acc = batch_acc / len(dataloaders[phase].dataset)

            # all_results[f"{phase}_loss"].append(epoch_loss)
            # all_results[f"{phase}_acc"].append(epoch_acc)

            utils.log(" {} Loss: {:.3f} Acc: {:.3f}.".format(phase, epoch_loss, epoch_acc))

            # # saving best model
            # if phase == "val" and epoch_acc >= best_val_acc:
            #     best_val_loss = epoch_loss
            #     best_val_acc = epoch_acc
            #     best_train_loss = all_results["train_loss"][-1]
            #     best_train_acc = all_results["train_acc"][-1]
            #     best_model_weights = copy.deepcopy(model.state_dict())
            #     utils.log(" saving model.")

        print()
        test(dataloaders["train"], model, criterion, exp_name)
        test(dataloaders["val"], model, criterion, exp_name)

    # utils.save_model(exp_name, best_model_weights)
    # utils.save_metrics(exp_name, best_train_loss, best_train_acc, best_val_loss, best_val_acc)
    # utils.plot_results(exp_name, all_results)