from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.nn as nn
import time
import copy
import os

from torch.utils.data import dataloader



def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=1).sum(dim=1)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + smooth)))
    
    return loss.mean()



def calc_loss(pred, target, metrics):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    metrics['bce loss'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice loss'] += dice.data.cpu().numpy() * target.size(0)
    
    return bce



def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    
    print(f'{phase}: {", ".join(outputs)}')



def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs=50, device='cuda'):
    best_loss = 100.

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to train mode
                dataloader = train_loader
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['bce'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("=> Saving checkpoint")
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model.pth'))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        scheduler.step()
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model