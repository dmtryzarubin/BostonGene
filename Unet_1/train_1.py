from os import name
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from dataset_utils import Segmentation_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from collections import defaultdict
import torch.nn.functional as F



def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)



def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])



def dice_loss(input, taget):
    print(input.size())
    print(taget.size())
    smooth=.001
    input=input.view(-1)
    target=taget.view(-1)
    
    return(1-2*(input*target).sum()/(input.sum()+taget.sum()+smooth))



def check_accuracy(loader, model, device='cuda'):
    bce = 0
    dice = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            bce = F.binary_cross_entropy_with_logits(model(x), y)
            preds = torch.sigmoid(model(x))
            dice = dice_loss(preds, y)
    
    print(
        f'\nval BCE: {bce:.4f}'
    )
    print(
        f'val dice: {dice:.4f}'
    )



def calc_acc(pred, target):
    probs = torch.log_softmax(pred, dim = 1)
    _, tags = torch.max(probs, dim=1)
    num_correct = torch.eq(tags, target).int().sum(dim=2)
    num_pixels = torch.numel(pred)

    return num_correct/num_pixels



def train_fn(loader, model, optimizer, scheduler, loss_fn, scaler, device='cuda'):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device) 
        
        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            dice = dice_loss(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer) 
        scaler.update()

        # Update tqdm Loop
        loop.set_postfix(loss=loss.item(), dice=dice.item())
    scheduler.step()