
import os
import numpy as np
import torch
import torch.nn
from PIL import Image
import data_utils
from tqdm import tqdm
import shutil
from torch.utils.data import DataLoader



def dice_coef(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()  

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    dice = (((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return dice.mean()


def reverse_transform(img):
    img = img.numpy().transpose([1, 2, 0])
    img = (img * 255).astype(np.uint8)
    return img


def calc_mean_dist(data_set):
    pass



def evaluate(model, loss_fn, test_dataset, batch_size, path, device='cuda'):
    """
    :param model: model to evaluate
    :param test_loader: test data loader
    :param path: path (where to save pretictions)
    :param device: device
    """
    if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    model.eval()
    metrics = {
        'bce' : [],
        'dice' : []
    }

    idx = 0
    for input, target in tqdm(test_loader):
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)

            metrics['bce'].append(loss_fn(model(input), target).cpu().numpy())
            preds = torch.sigmoid(model(input))
            metrics['dice'].append(dice_coef(preds, target).cpu().numpy())

            preds = preds.data.cpu().numpy()
            target = target.data.cpu().numpy()

        input_images_rgb = [reverse_transform(x) for x in input.cpu()]
        target_masks_rgb = [data_utils.masks_to_colorimg(x) for x in target]
        pred_rgb = [data_utils.masks_to_colorimg(x) for x in preds]

        # for mas, pred in zip(target_masks_rgb, pred_rgb):
        #     idx += 1
        #     ms = Image.fromarray(mas)
        #     pr.save(os.path.join(path, f'predicted_mask_{idx}.png'))
        #     pr = Image.fromarray(pred)
        #     pr.save(os.path.join(path, f'predicted_mask_{idx}.png'))
        
        torch.cuda.empty_cache()
    
    dice = np.average(metrics['dice'])
    bce = np.average(metrics['bce'])
    print(
        f'Dice score on test: {dice:.4f}\
            \nBCE loss on test: {bce:.4f}'
    )

    data_utils.plot_side_by_side([input_images_rgb[:3], target_masks_rgb[:3], pred_rgb[:3]])