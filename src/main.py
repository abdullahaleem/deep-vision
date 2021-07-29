import torch

from dataloaders import classification_dataloader
from networks import classification_networks, custom_networks
from train import train
from test import test
from config import config
import utils

exp_name = config["exp_name"]
model_name = config["model_name"]
in_channels = config["in_channels"]
out_features = config["out_features"]
pretrained = config["pretrained"]
learning_rate = config["learning_rate"]

utils.set_seeds(0)

dataloaders = classification_dataloader.create_dataloader(batch_size=80)
#utils.visualize_dataloader_classificaton(dataloaders["train"], exp_name)

model = classification_networks.resnet50(in_channels=1, out_features=1, pretrained=True)
model = custom_networks.AleemNet(in_channels=1, out_features=1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(dataloaders, model, optimizer, criterion, exp_name)