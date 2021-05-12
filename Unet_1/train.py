from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping



def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()



def calc_loss(pred, target, metrics, bce_weight=0.4):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce loss'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice loss'] += dice.data.cpu().numpy() * target.size(0)
    metrics['composite loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss



def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    
    print(f'{phase}: {", ".join(outputs)}')



def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs=50, patience=5, device='cuda'):
    best_loss = 1e3
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        since = time.time()
        metrics = defaultdict(float)
        epoch_samples = 0

        # Print learning rate
        for param_group in optimizer.param_groups:
                    print("LR:", param_group['lr'])
        
        # Training
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) 
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = calc_loss(outputs, labels, metrics)

            loss.backward()
            optimizer.step()
            epoch_samples += inputs.size(0)
        
        print_metrics(metrics, epoch_samples, 'Train')
        epoch_loss = metrics['composite loss'] / epoch_samples
        train_losses.append(epoch_loss)
        
        # Validation
        metrics = defaultdict(float)
        epoch_samples = 0

        model.eval()
        for inputs, labels in val_loader:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = calc_loss(outputs, labels, metrics)

            epoch_samples += inputs.size(0)

        print_metrics(metrics, epoch_samples, 'Val')
        epoch_loss = metrics['composite loss'] / epoch_samples
        val_losses.append(epoch_loss)
        
        if epoch_loss < best_loss:
                print("=> Saving model")
                best_loss = epoch_loss
                torch.save(model, os.path.join(os.getcwd(), 'model.pth'))
        
        early_stopping(epoch_loss, model)

        if early_stopping.early_stop:
            print('---Early Stop---')
            break

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        scheduler.step()

    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, train_losses, val_losses

def plot_loss(train_loss, valid_loss):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')