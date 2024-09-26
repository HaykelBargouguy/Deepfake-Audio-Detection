import time

import torch
import torch.nn.functional as F

from averager import Averager





def validate(val_loader, model, epoch,device, loss_func):
    avg = Averager()

    model.eval()

    with torch.no_grad():
        for idx ,(input,target) in enumerate(val_loader):
            input = input.to(device)
            target = target.view(-1).type(torch.int64).to(device)

            preds = model(input)
            loss = loss_func(preds, target)
            # compute loss
            avg.update_acc(preds, target)
            avg.update_loss(loss.item(), input.size(0))
            if idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{idx * len(input)}/{len(val_loader.dataset)}'
                      f' ({100. * idx / len(val_loader):.0f}%)]\tLoss: {loss.item():.6f}')


    accuracy, loss =avg.measure()
    avg.report("++> Validate {}".format(epoch))

    return accuracy,loss