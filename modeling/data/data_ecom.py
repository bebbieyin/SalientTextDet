import random
import os

import torch
from torch.utils.data import Dataset

from PIL import Image 
from modeling.data.generate_mask_bbox import LabelGeneration
random.seed(1)

globaltest = []

def collate_fn(batch):
    return tuple(zip(*batch)) 

def ecommercedata(data_path, img_nums, is_train, include_saliency, augment):
    global globaltest
    nums = [i for i in range(1, img_nums + 1)]
    if not globaltest:
        globaltest = random.sample(nums, img_nums // 10)
    test = globaltest
    train = list(set(nums) - set(test))

    if is_train:
        return EcommerceDataset(data_path=data_path, img_nums=(img_nums - (img_nums // 10)),
                                transform=augment, lis=train, include_saliency=include_saliency)
    else:
        return EcommerceDataset(data_path=data_path, img_nums=(img_nums // 10),
                                transform=augment, lis=test, include_saliency=include_saliency)
    
class EcommerceDataset(Dataset):

    def __init__(self, data_path, img_nums, transform, lis, include_saliency):
        """
        Args:
            data_path (string): Path to the imgs file with saliency.
            img_nums (int): Total number of images to index.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.transform = transform
        self.data_len = img_nums
        self.lis = lis
        self.text_label = LabelGeneration()
        self.saliency = include_saliency


    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_num = self.lis[idx]
        img_name = f'ALLSTIMULI/{sample_num}.jpg' 
        saliency_name = f'ALLFIXATIONMAPS/{sample_num}_fixMap.jpg'
        text_name = f'TEXT/gt_{sample_num}.txt'
        img_file = os.path.join(self.root_dir, img_name)
        saliency_file = os.path.join(self.root_dir, saliency_name)
        text_file = os.path.join(self.root_dir, text_name)

        # images
        img = Image.open(img_file).convert("RGB")
        
        # saliency
        saliency = Image.open(saliency_file).convert("L")

        #text
        boxes = self.text_label.get_annotations(text_file)
        text_mask =   self.text_label.box2mask(img,boxes)
        text_mask = Image.fromarray(text_mask).convert("L")

        
        torch_img = self.transform(img)
        torch_saliency = self.transform(saliency)
        torch_text_mask =  self.transform(text_mask)

        if self.saliency:
            return sample_num, torch_img, torch_saliency, torch_text_mask
        else:
            return sample_num, torch_img, torch_text_mask

