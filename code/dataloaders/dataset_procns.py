import itertools
import os
import random
import re
from collections import defaultdict
from glob import glob
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, num=None,sup_type="label",img_class='odoc'):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.img_class=img_class
        self.sup_type = sup_type
        self.transform = transform
        train_ids, val_ids = self._get_img_class_ids(img_class)
        if self.split == 'train':
            self.sample_list = train_ids
            print("total {} samples".format(len(self.sample_list)))

            self.images = defaultdict(dict)
            for idx, case in enumerate(self.sample_list):
                h5f = h5py.File(self._base_dir +"/{}".format(case), 'r')
                img = h5f['image']
                mask = h5f['mask']
                sup_label = h5f[self.sup_type]
                self.images[idx]['id'] = case
                self.images[idx]['image'] = np.array(img)
                self.images[idx]['mask'] = np.array(mask)
                self.images[idx]['gt'] = np.array(mask)
                self.images[idx][self.sup_type] = np.array(sup_label)
                h, w = mask.shape
                # self.images[idx]['weight'] = np.zeros((h, w, 4), dtype=np.float32)
                if img_class=='odoc':
                    self.images[idx]['weight'] = np.zeros((3, h, w), dtype=np.float32)
                if img_class=='faz' or img_class == 'polyp':
                    self.images[idx]['weight'] = np.zeros((2, h, w), dtype=np.float32)
        elif self.split == 'val':
            self.sample_list=val_ids      
        print("total {} samples".format(len(self.sample_list)))
            

    def _get_img_class_ids(self,image_class):
        
        if image_class == "faz":
            faz_test_set = 'FAZ/test/'+pd.Series(os.listdir(self._base_dir+"/FAZ/test"))
            faz_training_set = 'FAZ/train/'+pd.Series(os.listdir(self._base_dir+"/FAZ/train"))
            faz_test_set = faz_test_set.tolist()
            faz_training_set = faz_training_set.tolist()
            return [faz_training_set, faz_test_set]
        elif image_class == "odoc":
            odoc_test_set = 'ODOC/test/'+pd.Series(os.listdir(self._base_dir+"/ODOC/test"))
            odoc_training_set = 'ODOC/train/'+pd.Series(os.listdir(self._base_dir+"/ODOC/train"))
            odoc_test_set = odoc_test_set.tolist()
            odoc_training_set = odoc_training_set.tolist()
            return [odoc_training_set, odoc_test_set]
        elif image_class == "polyp":
            polyp_test_set = 'Polyp/test/'+pd.Series(os.listdir(self._base_dir+"/Polyp/test"))
            polyp_training_set = 'Polyp/train/'+pd.Series(os.listdir(self._base_dir+"/Polyp/train"))
            polyp_test_set = polyp_test_set.tolist()
            polyp_training_set = polyp_training_set.tolist()
            return [polyp_training_set, polyp_test_set]
        
        else:
            return "ERROR KEY"
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            case = self.images[idx]['id'][:]
            image = self.images[idx]['image'][:]
            gt = self.images[idx]['gt'][:]
            mask = self.images[idx]['mask'][:]
            sup_label = self.images[idx][self.sup_type][:]
            weight = self.images[idx]['weight'][:]
            sample = {'image': image, 'mask': mask,
                    'sup_label': sup_label, 'weight': weight, 'gt':gt}
            sample = self.transform(sample)
            sample['id'] = case
        if self.split == 'val':
            case = self.sample_list[idx]
            h5f = h5py.File(self._base_dir +
                            "/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['mask'][:]
            sample = {'image': image, 'label': label}
            sample["id"] = idx

        return sample


def random_rot_flip(image, label, sup_label, weight, img_class='odoc'):
    if img_class == 'odoc' or img_class == 'polyp':
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, (1,2))
        label = np.rot90(label, k, (0,1))
        weight = np.rot90(weight, k,(1,2))
        sup_label = np.rot90(sup_label, k,(0,1))
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis-1).copy()
        weight = np.flip(weight, axis=axis).copy()
        sup_label = np.flip(sup_label, axis=axis-1).copy()
        
        return image, label, sup_label, weight
    if img_class == 'faz':
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        sup_label = np.rot90(sup_label, k)
        weight = np.rot90(weight, k, (1,2))
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        sup_label = np.flip(sup_label, axis=axis).copy()
        weight = np.flip(weight, axis=axis+1).copy()
        return image, label, sup_label, weight


def random_rotate(image, label, sup_label, weight, img_class='odoc'):
    if img_class=='faz':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, order=0, reshape=False, cval=0.8)
        label = ndimage.rotate(label, angle, order=0,
                            reshape=False, mode="constant", cval=2)
        sup_label = ndimage.rotate(sup_label, angle, order=0, reshape=False,cval=2)
        weight = ndimage.rotate(weight, angle, axes=(1,2), order=0, reshape=False,cval=0)
        return image, label, sup_label, weight
    if img_class=='odoc':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, axes=(1,2), order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=(0,1), order=0,reshape=False, mode="constant")
        sup_label = ndimage.rotate(sup_label, angle,axes=(0,1), order=0, reshape=False,cval=3)
        weight = ndimage.rotate(weight, angle, axes=(1,2),order=0, reshape=False)
        return image, label, sup_label, weight
    if img_class=='polyp':
        angle = np.random.randint(-45, 45)
        image = ndimage.rotate(image, angle, axes=(1,2), order=0, reshape=False)
        label = ndimage.rotate(label, angle, axes=(0,1), order=0,reshape=False, mode="constant")
        sup_label = ndimage.rotate(sup_label, angle,axes=(0,1), order=0, reshape=False,cval=2)
        weight = ndimage.rotate(weight, angle, axes=(1,2),order=0, reshape=False)
        return image, label, sup_label, weight

class RandomGenerator(object):
    def __init__(self, output_size,img_class):
        self.output_size = output_size
        self.img_class = img_class

    def __call__(self, sample):
        image, label, sup_label, weight = sample['image'], sample['mask'], sample['sup_label'], sample['weight']
        if random.random() > 0.5:
            image, label, sup_label, weight = random_rot_flip(
                image, label, sup_label, weight,img_class=self.img_class)
        if random.random() > 0.5:
            image, label, sup_label, weight = random_rotate(
                image, label, sup_label, weight,img_class=self.img_class)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sup_label = torch.from_numpy(sup_label.astype(np.uint8))
        weight = torch.from_numpy(weight.astype(np.float32))
        sample = {'image': image, 'mask': label,
                  'sup_label': sup_label, 'weight': weight, 'gt':label}
        return sample




