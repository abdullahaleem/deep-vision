import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_model(exp_name, weights):
    model_dir = f"../output/models/"
    make_dir(model_dir)
    torch.save(weights, os.path.join(model_dir, exp_name+".pth"))

def load_model_weights(exp_name, model):
    model_dir = f"../output/models/"
    model.load_state_dict(torch.load(os.path.join(model_dir, exp_name+".pth")))
    return model


def save_metrics(exp_name, train_loss, train_acc, val_loss, val_acc):
    metric_dir = f"../output/metrics/"
    make_dir(metric_dir)
    metric_filename = os.path.join(metric_dir, f"{exp_name}_train.csv")
    
    with open(metric_filename, "a+") as f:
        to_write = f"{exp_name}, {train_loss:.3f}, {train_acc:.3f}, {val_loss:.3f}, {val_acc:.3f}"
        print(to_write, file=f)


def log(statement):
    # log_path = rf"../output/logs/",
    # make_dir(log_path)
    # log_file = os.path.join(log_path, model_name + ".txt")

    sys.stdout.write(statement)
    sys.stdout.flush()
    # f = open(path_file, 'a')
    # f.write(statement)
    # f.close()

def plot_results(exp_name, all_results):
    
    plot_dir = f"../output/plots/"
    make_dir(plot_dir)

    train_accuracies = all_results["train_acc"]
    val_accuracies = all_results["val_acc"]
    train_losses = all_results["train_loss"]
    val_losses = all_results["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(20, 8)

    # plotting accuracy
    ax1.plot(range(len(train_accuracies)), train_accuracies, label = "train")
    ax1.plot(range(len(val_accuracies)), val_accuracies, label = "val")
    ax1.set_xlim(0)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend()
    #plt.axvline(best_model_epoch_number, color="black")
    #plt.axhline(best_acc, color="black")

    # plotting loss
    ax2.plot(range(len(train_losses)), train_losses, label = "train")
    ax2.plot(range(len(val_losses)), val_losses, label = "val")
    ax2.set_xlim(0)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend()

    fig.savefig(os.path.join(plot_dir, exp_name + ".png"))



def visualize_dataloader_classificaton(dataloader, exp_name):
    visualize_path = rf"../output/visualize/{exp_name}/datalaoder"
    make_dir(visualize_path)

    counter = 0
    idx_to_class = {value : key for (key, value) in dataloader.dataset.class_to_idx.items()}

    for batch_num, (inputs, targets) in enumerate(dataloader):
        for image_num, image in enumerate(inputs):  
            image = image.numpy()*255
            image = np.moveaxis(image, 0, 2)
            target = int(targets[image_num].numpy())
            target = idx_to_class[target]
            
            counter += 1
            cv2.imwrite(os.path.join(visualize_path, f"{target}{counter}.png"), image)
            #cv2.imwrite(os.path.join(visualize_path, f"batch{batch_num}_image{image_num}_{target}.png"), image)


def visualize_dataloader_segmentation(dataloader, exp_name):
    visualize_path = rf"../output/visualize/{exp_name}/datalaoder"
    make_dir(visualize_path)

    counter = 0
    idx_to_class = {value : key for (key, value) in dataloader.dataset.class_to_idx.items()}

    pass


