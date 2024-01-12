import random
import os

import torch
from torch.utils.data import Dataset

from PIL import Image 
from modeling.data.generate_mask_bbox import LabelGeneration

random.seed(1)

# get sample numbers from icdar15 files
def icdar15_sample_num(directory_path):

    img_numbers = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") and filename.startswith("img_"):
            # Split the filename by '_' and get the second part
            parts = filename.split("_")
            if len(parts) == 2:
                img_number = parts[1].split(".")[0]
                img_numbers.append(int(img_number))
    return img_numbers
            
def icdar15data(data_path, img_nums, is_train, augment,ignore_donotcare):

    if is_train:
        data_dir = os.path.join(data_path,'Train')
        train = icdar15_sample_num(data_dir)
        
        return ICDAR15Dataset(data_path=data_dir, img_nums=img_nums,
                            transform=augment, lis=train,img_size=896,ignore_donotcare=ignore_donotcare) 
    else:
        data_dir = os.path.join(data_path,'Test')
        test =  icdar15_sample_num(data_dir)
        return ICDAR15Dataset(data_path=data_dir, img_nums=img_nums,
                            transform=augment, lis=test,img_size=896,ignore_donotcare=ignore_donotcare) 
        
class ICDAR15Dataset(Dataset):

    def __init__(self, data_path, img_nums, transform, lis, img_size, ignore_donotcare):
        """
        Args:
            data_path (string): Path to the imgs file with text annotations.
            img_nums (int): Total number of images to index.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.transform = transform
        self.data_len = img_nums
        self.lis = lis
        self.img_size = img_size
        self.text_label = LabelGeneration()
        self.ignore_donotcare = ignore_donotcare

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_num = self.lis[idx]
        img_name = f'img_{sample_num}.jpg' 
        gt_name = f'gt_img_{sample_num}.txt'
        
        img_file = os.path.join(self.root_dir, img_name)
        gt_file = os.path.join(self.root_dir, gt_name)

        # images
        img = Image.open(img_file).convert("RGB")

        #text
        boxes = self.text_label.get_annotations_icdar15(gt_file,remove_donotcare=self.ignore_donotcare)
        text_mask =   self.text_label.box2mask(img,boxes)
        text_mask = Image.fromarray(text_mask).convert("L")

        torch_img = self.transform(img)
        torch_text_mask =  self.transform(text_mask)

        return sample_num, torch_img, torch_text_mask # no saliency gt 
    
    