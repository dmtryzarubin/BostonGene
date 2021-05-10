import os
import shutil

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import generate_random_data, masks_to_colorimg



def create_dataset(height, width, n_train, n_val, n_test, root):
    """
    Creates a two-folder dataset with images and masks
    :params height, width: image height and width
    :param n_imgs: number of Images in the dataset
    :param ROOT: Root folder
    """
    folder = 'Segmentation_Data'
    path = os.path.join(root, folder)
    sets = ['Train', 'Val', 'Test']
    subfolders = ['Images', 'Masks']
    n = [n_train, n_val, n_test] 

    def generate_n_save(path, n_imgs):
        """
        Generating n images (n_imgs) and parsing them into folders.
        """
        for n in tqdm(range(1, n_imgs + 1)):
            img, masks = generate_random_data(height, width, 1)

            
            # Convert Image to RGB
            img_rgb = [x.astype(np.uint8) for x in img][0]
            # Each objects has its intensity. Ex: (index = 1 -> intensity = 1, etc)
            # Get single channel
            mask = np.max(masks[0], axis=0) 

            im = Image.fromarray(img_rgb)
            im.save(os.path.join(path, 'Images') + '\\img_' + str(n) + '.png')
            ms = Image.fromarray(mask).convert("P")
            ms.save(os.path.join(path, 'Masks') + '\\mask_' + str(n) + '.png')
    
    # Checking if dir already exists
    if os.path.exists(path):
            print(f'Directory |{path}|\nAlready exists! It will be removed')
            shutil.rmtree(path, ignore_errors=True)

    # Creating two directories: for masks and images.
    
    try:
        os.makedirs(path)
        for idx, s in enumerate(sets):
            subset_path = os.path.join(path, s)
            os.makedirs(subset_path)
            for subfolder in subfolders:
                subfolder_path = os.path.join(subset_path, subfolder)
                os.makedirs(subfolder_path)
            generate_n_save(subset_path, n[idx])
            
    except OSError:
        print (f'Creation of the directory |{path}| failed')




def plot_img_n_masks(root, num_imgs):
    """
    Plots saved images and masks
    :param root: Data root folder
    :param num_imgs: Number of images to plot
    """
    img_path = os.path.join(root,'Images')
    masks_path = os.path.join(root,'Masks')
    idxs = np.random.randint(low=1, high=len(os.listdir(img_path)), size=num_imgs)
    fig, axs = plt.subplots(nrows=num_imgs, ncols=num_imgs, figsize=(7, 7), sharex='all', sharey='all')

    for i in range(0, num_imgs):
        img = Image.open(os.path.join(img_path,f'img_{idxs[i]}.png'))
        mask = Image.open(os.path.join(masks_path,f'mask_{idxs[i]}.png'))
        
        img = np.array(img)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None] 
        masks = masks.astype('float32')

        axs[i, 0].imshow(img)
        axs[i, 1].imshow(masks_to_colorimg(masks))
    plt.suptitle('Images and Masks', fontsize="x-large")
    plt.tight_layout()
    plt.show()    




class Segmentation_Dataset(torch.utils.data.Dataset):
    """
    Reinitialized Segmentation dataset class
    """

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path), dtype='float32')

        reference = np.array([ 10., 20., 30., 40., 50., 60.])

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]


        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        masks = masks.astype('float32')

        # If some objects were hided by other ones => we just creating an empty mask for that object = There is no such kind of object on the image
        if len(obj_ids) != len(reference):
            missing_ch = list(set(reference) - set(obj_ids))
            for i in sorted(missing_ch):
                masks = np.insert(
                     masks, 
                     int(i // 10) - 1, 
                     np.zeros((mask.shape[0], mask.shape[1]), dtype='float32'),
                     axis=0)

        if self.transforms is not None:
            img = self.transforms(img)
            

        return img, masks

    def __len__(self):
        return len(self.imgs)