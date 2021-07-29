import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def create_dataloader(batch_size=32):
    transform = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomAffine(degrees=(-10,10), translate=(0.10, 0.10)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
    }

    # Loading data from folders
    data_dir = "../data/classification"
    train_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform["test"])
    val_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform["test"])
    test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform["test"])

    # # Print data specifications
    # print("---Data Specifications---")
    # print("train data length:", len(train_data))
    # print("val data length:", len(val_data))
    # print("test data length:", len(test_data))
    # print("train data classes:", train_data.class_to_idx)
    # print("val data classes:", val_data.class_to_idx)
    # print("test data classes:", test_data.class_to_idx)

    # Creating a balanced sampler for training data
    train_classes = train_data.classes
    train_classes_count = [train_data.targets.count(train_data.class_to_idx[train_class]) for train_class in train_classes]
    train_classes_ratio = (1 / torch.Tensor(train_classes_count)).double()
    train_classes_weights = np.array([train_classes_ratio[target] for target in train_data.targets])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_classes_weights, len(train_classes_weights))

    # Creating data loader for train, val and test sets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return {"train":train_loader, "val": val_loader, "test": test_loader}
