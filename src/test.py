import torch
import utils
from config import config


def test(dataloader, model, criterion, exp_name, wandb=None):

    #model = utils.load_model_weights(exp_name, model)
    model.eval()

    epoch_loss, epoch_acc = 0, 0
    batch_acc, batch_loss = 0, 0

    for inputs, targets in dataloader:
        if torch.cuda.is_available(): 
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)                     
            targets = targets.unsqueeze(1).float()

            loss = criterion(outputs, targets)
            predictions = torch.round(outputs)
            #_, predictions = torch.max(outputs, 1)

        batch_loss += loss.item() * len(inputs)
        batch_acc += torch.mean((predictions == targets) * 1.0) * len(inputs)

    epoch_loss = batch_loss / len(dataloader.dataset)
    epoch_acc = batch_acc / len(dataloader.dataset)

    utils.log("Test Loss: {:.3f} Acc: {:.3f}.".format(epoch_loss, epoch_acc))