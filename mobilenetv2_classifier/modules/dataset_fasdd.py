import os
from pathlib import Path
import numpy as np
import math
import random
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class FASDDDataset(Dataset):
    '''
    Creates a Pytorch Dataset to train the BED Classifier.
    
    Arguments:
        - img_h:            image height
        - img_w:            image width
        - imgs_dir:         path to images folder
        - labels_dir:       path to labels folder
        - num_classes:      number of classes
        - transform:        transformation applied to input images -> Albumentations
        - target_transform: transformation applied to labels -> nothing by default

    Return:
        - img:              1 image of the dataset
        - target:           corresponding label encoded: [smoke, fire]
    '''

    def __init__(self, img_h, img_w, imgs_dir, labels_file, num_classes,
                 ds_len,
                 transform=None, target_transform=None,
                 return_img_filename=False):
        self.img_h = img_h
        self.img_w = img_w
        self.imgs_dir = imgs_dir
        self.labels_file = labels_file
        self.num_classes = num_classes
        self.ds_len = ds_len
        self.transform = transform
        self.target_transform = target_transform
        self.return_img_filename = return_img_filename
        
        self.except_transform = A.Compose([
                                    A.Resize(self.img_h, self.img_w, p=1),
                                    ToTensorV2(p=1),
                                    ]
                                )

        self.labels_list = self.__get_labels_list__()
    
        if ds_len is not None:
            random.seed(123)
            random.shuffle(self.labels_list)
            self.images_path, self.labels = self.__build_ds__(self.labels_list[:self.ds_len])
            self.num_samples = len(self.labels_list[:self.ds_len])
        else:
            self.images_path, self.labels = self.__build_ds__(self.labels_list)
            self.num_samples = len(self.labels_list)

    def __len__(self):
        return self.num_samples  

    def __get_labels_list__(self):
        labels_list = []
        with open(self.labels_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('/')[-1]
                labels_list.append(self.imgs_dir + line)
        return labels_list
            

    def __build_ds__(self, labels_list):
        labels = []
        images = []
        wrong_imgs = 0
        empty = 0
        only_smoke = 0
        only_fire = 0
        smoke_fire = 0
                
        for image_path in labels_list:
            fname = Path(image_path).stem
            #print(fname, image_path)
                                   
            if cv2.imread(image_path) is None:
                print(f'{image_path} cannot be read by cv2 -> removed')
                wrong_imgs += 1
            
            else: 
                if 'both' in fname:
                    label_array = np.array([1, 1])
                    smoke_fire += 1
                elif 'neither' in fname:
                    label_array = np.array([0, 0])
                    empty += 1
                elif 'smoke' in fname:
                    label_array = np.array([1, 0])
                    only_smoke += 1
                elif 'fire' in fname:
                    label_array = np.array([0, 1])
                    only_fire += 1
                else:
                    print("Wrong Label")
                        
                labels.append(label_array)
                images.append(image_path)
        
        print(f'FASDD Removed wrong images: {wrong_imgs}')
        print(f'FASDD empty images: {empty}')
        print(f'FASDD only smoke images: {only_smoke}')
        print(f'FASDD only fire images: {only_fire}')
        print(f'FASDD smoke and fire images: {smoke_fire}')

        labels_np = np.array(labels)
        labels_tensor = torch.tensor(labels_np, dtype=torch.float32)
        images_array = np.array(images)
        
        return images_array, labels_tensor

    def __getitem__(self, index):

        # Image processing
        img_file = self.images_path[index]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   

        # Labels processing
        label = self.labels[index]
        
        # Data Augmentation
        if self.transform is not None:
            try:
                aug = self.transform(image=img)
                img = aug['image'] / 255.0 #256.0 #255.0
            except:
                #print(f'Error trying to augment image {img_file}')
                aug = self.except_transform(image=img)
                img = aug['image'] / 255.0 #256.0 #255.0
        
        if self.return_img_filename:
            return img, label, img_file    
        else:
            return img, label