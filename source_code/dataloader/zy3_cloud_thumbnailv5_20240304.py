'''
Have modified the code to adapt to the new data format of zy3 cloud dataset.
Now increase the training samples with 224x224x3 size.
the test samples with 224x224x3 size from 50.

Created by ZhouYi@Linghai_Dalian on 2023/12/26

Add the augmentation transforming to the training set.
Add the dataset class with supervised and un-supervised options.

Modified by ZhouYi@Linghai_Dalian on 2024/03/05
Add the dataset class for the bright terrains in trainset with added clouds.
In the trainset, we use the cloudless bright terrains (inc. water, snow, ice, sand, etc.) as the background,
and add the clouds to the background to generate the cloud images.


Modified by ZhouYi@Linghai_Dalian on 2024/01/22
Upgrated by ZhouYi@Linghai_Dalian on 2024/01/26
Upgrated by ZhouYi@Linghai_Dalian on 2024/03/04
'''

import os
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
import argparse
from glob import glob
from tqdm import tqdm
import sys
#from skimage.color import rgb2gray
import gc
import utils_20231218 as uti
import torchvision.transforms as transforms
import pickle
import albumentations as A


alb_augmentation = A.Compose([
                    A.OneOf([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Transpose(p=0.5),
                        A.Perspective(always_apply=False, scale=(0.05, 0.1), keep_size=True, pad_mode=0,
                                      pad_val=(0, 0, 0), mask_pad_val=0, fit_output=0, interpolation=0, p=0.1),
                    ], p=0.8),
                    A.Rotate(always_apply=False, limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0),
                             mask_value=None, rotate_method='largest_box', crop_border=False, p=0.2),
                    A.RandomSnow(always_apply=False,  snow_point_lower=0.1, snow_point_upper=0.2, brightness_coeff=2.5,p=0.1),
                    A.OneOf([
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.1),
                        A.GridDistortion(p=0.1),
                        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.1),
                        A.Defocus(always_apply=False, radius=(3, 10), alias_blur=(0.1, 0.5), p=0.2),
                        A.Emboss(always_apply=False, alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.1),
                        A.GridDistortion(always_apply=False, num_steps=5, distort_limit=(-0.3, 0.3),
                                         interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None,
                                         normalized=False, p=0.1),
                        #A.JpegCompression(always_apply=False, quality_lower=80, quality_upper=100, p=0.2),
                        A.ElasticTransform(always_apply=False, alpha=1.0, sigma=50.0, alpha_affine=50.0,
                                           interpolation=0, border_mode=0, value=(0, 0, 0),
                                           mask_value=None, approximate=False, same_dxdy=False, p=0.1),
                    ], p=0.1),
                    #A.CLAHE(p=0.2), #uint8 only
                    A.RandomBrightnessContrast(always_apply=False,
                                               brightness_limit=(0.04, 0.38),
                                               contrast_limit=(-0.19, 0.35),
                                               brightness_by_max=True,
                                               p=0.1),
                    # A.CoarseDropout(always_apply=False, max_holes=8, max_height=8, max_width=8, min_holes=8,
                    #                 min_height=8, min_width=8, fill_value=(0, 0, 0), mask_fill_value=None, p=0.2),
                    #
                    # A.PixelDropout(always_apply=False,  dropout_prob=0.01, per_channel=0, drop_value=(0, 0, 0), mask_drop_value=None, p=0.3),
                ], p=1.)
def prepare_cloud_traindata(config):
    '''
    Load the source images
    :param config:
    :return:
    '''
    config.train_file = os.path.join(config.dataset_root, config.train_file)
    assert os.path.exists(config.train_file), 'the train file %s in %s does not exist!'%(config.train_file, config.dataset_root)
    print('%s file exists, load it directly.' % config.train_file)
    if 'pkl' in config.test_file:
        with open(config.train_file, 'rb') as f:
            img_train_dict = pickle.load(f)
    if '.pt' in config.train_file:
        img_train_dict = torch.load(config.train_file)

    return img_train_dict

def prepare_cloud_testdata(config):
    config.test_file = os.path.join(config.dataset_root, config.test_file)
    assert os.path.exists(config.test_file), 'the test file %s in %s does not exist!'%(config.test_file, config.dataset_root)
    print('%s file exists, load it directly.' % config.test_file)
    if 'pkl' in config.test_file:
        with open(config.test_file, 'rb') as f:
            img_test_label_dict = pickle.load(f)
    if '.pt' in config.test_file:
        img_test_label_dict = torch.load(config.test_file)
    return img_test_label_dict

def prepare_added_cloud_data(path):
    '''
    Load the source images
    :param config:
    :return:
    '''
    #config.train_file = os.path.join(config.dataset_root, config.train_file)
    assert os.path.exists(path), 'the source data file %s does not exist!'%path
    print('%s file exists, load it directly.' % path)
    if 'pkl' in path:
        with open(path, 'rb') as f:
            img_dict = pickle.load(f)
    if '.pt' in path:
        img_dict = torch.load(path)
    print(''.join(['The number of samples in the dataset is ', str(len(img_dict))]))
    return img_dict
class CloudDataset_Supervised(Dataset):
    '''
    Refer to https://www.kaggle.com/code/cordmaur/38-cloud-data-preparation
    Context
    This dataset contains 38 Landsat 8 scene images and their manually extracted pixel-level ground truths for cloud detection.
    Content
    The entire images of these scenes are cropped into multiple 384*384 patches
    to be proper for deep learning-based semantic segmentation algorithms.
    There are 8400 patches for training and 9201 patches for testing.
    Each patch has 4 corresponding spectral channels which are Red (band 4), Green (band 3), Blue (band 2),
    and Near Infrared (band 5). Unlike other computer vision images, these channels are not combined.
    Instead, they are in their corresponding directories.
    '''
    def __init__(self, data_dict, baug=True):
        super().__init__()
        self.image_label_dict = data_dict # testset has rgb and labels.
        self.image_ids = list(data_dict.keys())
        self.baug = baug # whether to do the augmentation.

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        key = self.image_ids[idx]
        image_id = key
        if self.baug: # augmentation is conducted on images and labels both.
            raw_rgb =  self.image_label_dict[key]['true_color']
            label    = self.image_label_dict[key]['mask']
            #albumentation augmentation use numpy as input.
            rgb_array = raw_rgb.permute(1,2,0).numpy() # [h, w, 3]
            label_array = label.numpy() # [h, w]
            augmented = alb_augmentation(image=rgb_array, mask=label_array)
            aug_rgb = augmented['image']
            aug_label = augmented['mask']
            aug_rgb = torch.tensor(aug_rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, h, w]
            aug_label = torch.tensor(aug_label, dtype=torch.float32)
            return aug_rgb, aug_label, image_id

        else: #test get gray, rgb image and labels
            raw_rgb =  self.image_label_dict[key]['true_color']
            #raw_rgb  = torch.tensor(raw_rgb, dtype=torch.float32)
            #raw_rgb  = raw_rgb.permute(2, 0, 1) #[3, h, w]
            label    = self.image_label_dict[key]['mask']
            #label    = torch.tensor(label, dtype=torch.float32)
            return  raw_rgb, label, image_id

    def __repr__(self):
        s = 'Supervised Dataset class with {} files'.format(self.__len__())
        return s

class Snow_CloudDataset_Supervised(Dataset):
    '''
    snow_cloud dataset with rgb and mask for clouds and snow_mask for snow.
    '''
    def __init__(self, data_dict, baug=True):
        super().__init__()
        self.image_label_dict = data_dict # testset has rgb and labels.
        self.image_ids = list(data_dict.keys())
        self.baug = baug # whether to do the augmentation.

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        key = self.image_ids[idx]
        image_id = key
        if self.baug: # augmentation is conducted on images and labels both.
            raw_rgb =  self.image_label_dict[key]['true_color']
            label    = self.image_label_dict[key]['mask']
            #albumentation augmentation use numpy as input.
            rgb_array = raw_rgb.permute(1,2,0).numpy() # [h, w, 3]
            label_array = label.numpy() # [h, w]
            augmented = alb_augmentation(image=rgb_array, mask=label_array)
            aug_rgb = augmented['image']
            aug_label = augmented['mask']
            aug_cloud_label = (aug_label==1)
            aug_snow_label  = (aug_label==2)
            aug_rgb = torch.tensor(aug_rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, h, w]
            aug_cloud_label = torch.tensor(aug_cloud_label, dtype=torch.float32)
            aug_snow_label  = torch.tensor(aug_snow_label, dtype=torch.float32)
            return aug_rgb, aug_cloud_label, aug_snow_label, image_id

        else: #test get gray, rgb image and labels
            raw_rgb =  self.image_label_dict[key]['true_color']
            #raw_rgb  = torch.tensor(raw_rgb, dtype=torch.float32)
            #raw_rgb  = raw_rgb.permute(2, 0, 1) #[3, h, w]
            label    = self.image_label_dict[key]['mask']
            #label    = torch.tensor(label, dtype=torch.float32)
            return  raw_rgb, label, image_id

    def __repr__(self):
        s = 'Supervised Dataset class with {} files'.format(self.__len__())
        return s

class CloudDataset_Unsupervised(Dataset):
    '''
    Refer to https://www.kaggle.com/code/cordmaur/38-cloud-data-preparation
    Context
    This dataset contains 38 Landsat 8 scene images and their manually extracted pixel-level ground truths for cloud detection.
    Content
    The entire images of these scenes are cropped into multiple 384*384 patches
    to be proper for deep learning-based semantic segmentation algorithms.
    There are 8400 patches for training and 9201 patches for testing.
    Each patch has 4 corresponding spectral channels which are Red (band 4), Green (band 3), Blue (band 2),
    and Near Infrared (band 5). Unlike other computer vision images, these channels are not combined.
    Instead, they are in their corresponding directories.
    '''
    def __init__(self, data_dict, baug=True):
        super().__init__()
        self.image_dict = data_dict # testset has rgb and labels.
        self.image_ids = list(data_dict.keys())
        self.baug = baug # whether to do the augmentation.

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        key = self.image_ids[idx]
        image_id = key
        if self.baug: # augmentation is conducted on images and labels both.
            raw_rgb =  self.image_dict[key]['true_color']
            #albumentation augmentation use numpy as input.
            rgb_array = raw_rgb.permute(1,2,0).numpy() # [h, w, 3]
            augmented = alb_augmentation(image=rgb_array)
            aug_rgb = augmented['image']
            aug_rgb = torch.tensor(aug_rgb, dtype=torch.float32).permute(2, 0, 1)  # [3, h, w]
            return aug_rgb, image_id

        else: #test get gray, rgb image and labels
            raw_rgb =  self.image_dict[key]['true_color']
            #raw_rgb  = torch.tensor(raw_rgb, dtype=torch.float32)
            #raw_rgb  = raw_rgb.permute(2, 0, 1) #[3, h, w]
            return  raw_rgb, image_id

    def __repr__(self):
        s = 'Unsupervised Dataset class with {} files'.format(self.__len__())
        return s

class CloudDataset_CloudAddition(Dataset):
    '''
    Refer to https://www.kaggle.com/code/cordmaur/38-cloud-data-preparation
    Context
    This dataset contains 38 Landsat 8 scene images and their manually extracted pixel-level ground truths for cloud detection.
    Content
    The entire images of these scenes are cropped into multiple 384*384 patches
    to be proper for deep learning-based semantic segmentation algorithms.
    There are 8400 patches for training and 9201 patches for testing.
    Each patch has 4 corresponding spectral channels which are Red (band 4), Green (band 3), Blue (band 2),
    and Near Infrared (band 5). Unlike other computer vision images, these channels are not combined.
    Instead, they are in their corresponding directories.
    '''
    def __init__(self, data_dict, baug=True):
        super().__init__()
        self.image_label_dict = data_dict # testset has rgb and labels.
        self.image_ids = list(data_dict.keys())
        self.baug = baug # whether to do the augmentation.

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        key = self.image_ids[idx]
        image_id = key
        if self.baug: # augmentation is conducted on images and labels both.
            raw_rgb =  self.image_label_dict[key]['terrain']
            syc_rgb =  self.image_label_dict[key]['true_color']
            label    = self.image_label_dict[key]['mask']
            #albumentation augmentation use numpy as input.
            syc_array   = syc_rgb.permute(1,2,0).numpy() # [h, w, 3]
            label_array = label.numpy() # [h, w]
            augmented   = alb_augmentation(image=syc_array, mask=label_array)
            aug_syc     = augmented['image']
            aug_label   = augmented['mask']
            aug_syc     = torch.tensor(aug_syc, dtype=torch.float32).permute(2, 0, 1)  # [3, h, w]
            aug_label   = torch.tensor(aug_label, dtype=torch.float32)
            return raw_rgb, aug_syc, aug_label, image_id

        else: #test get gray, rgb image and labels
            raw_rgb =  self.image_label_dict[key]['terrain']
            syc_rgb =  self.image_label_dict[key]['true_color']
            label    = self.image_label_dict[key]['mask']
            return  raw_rgb, syc_rgb, label, image_id

    def __repr__(self):
        s = 'Synthetic cloud Dataset class with {} files'.format(self.__len__())
        return s

def cloud_dataloader_imagelabel_dict(config):

    cloud_image_label_test_dict = prepare_cloud_testdata(config)      # load the test data with image and gt_label.
    cloud_image_train_dict      = prepare_cloud_traindata(config)
    train_data = CloudDataset_Unsupervised(cloud_image_train_dict,    baug=False)
    test_data  = CloudDataset_Supervised(cloud_image_label_test_dict, baug=False)
    train_dl   = DataLoader(train_data, batch_size=config.batch_sz,   shuffle=True)
    test_dl    = DataLoader(test_data, batch_size=config.batch_sz,    shuffle=False)

    return train_dl, test_dl, cloud_image_train_dict, cloud_image_label_test_dict


def cloud_dataloader(config):

    cloud_image_train_dict = prepare_cloud_traindata(config)
    cloud_image_label_dict = prepare_cloud_testdata(config)

    train_data = CloudDataset_Unsupervised(cloud_image_train_dict, baug=False) # iteration return rgb image and image_id, no labels
    test_data  = CloudDataset_Supervised(cloud_image_label_dict,   baug=False) # iteration return rgb image, label and image_id

    train_dl = DataLoader(train_data, batch_size=config.batch_sz,   shuffle=True)
    test_dl  = DataLoader(test_data,  batch_size=config.batch_sz,   shuffle=True)
    return train_dl, test_dl

def cloud_dataloader_cloud_addition(config):
    config.added_cloud_file    = os.path.join(config.dataset_root, config.added_cloud_file)
    cloud_add_image_label_dict = prepare_added_cloud_data(config.added_cloud_file)
    cloud_add_train_data       = CloudDataset_CloudAddition(cloud_add_image_label_dict, baug=config.aug)
    cloud_add_train_dl         = DataLoader(cloud_add_train_data, batch_size=config.batch_sz, shuffle=True)
    return cloud_add_train_dl, cloud_add_image_label_dict

def snow_cloud_dataloader_via_dict(config, image_label_dict, bsu=False, baug=False):
    '''
    :param config:
    :param image_label_dict:  the dict of the image and label
    :param bsu:  whether to use supervised learning
    :param baug: whether to do the augmentation
    :return:
    '''
    if bsu:
        test_data = Snow_CloudDataset_Supervised(image_label_dict, baug=baug) # generate image and label in each iteration.
        test_dl = DataLoader(test_data, batch_size=config.batch_sz, shuffle=True)
        return test_dl
    else:
        train_data = CloudDataset_Unsupervised(image_label_dict, baug=baug)
        train_dl = DataLoader(train_data, batch_size=config.batch_sz, shuffle=True)
        return train_dl

def cloud_dataloader_via_dict(config, image_label_dict, bsu=False, baug=False):
    '''
    :param config:
    :param image_label_dict:  the dict of the image and label
    :param bsu:  whether to use supervised learning
    :param baug: whether to do the augmentation
    :return:
    '''
    if bsu:
        test_data = CloudDataset_Supervised(image_label_dict, baug=baug) # generate image and label in each iteration.
        test_dl = DataLoader(test_data, batch_size=config.batch_sz, shuffle=True)
        return test_dl
    else:
        train_data = CloudDataset_Unsupervised(image_label_dict, baug=baug)
        train_dl = DataLoader(train_data, batch_size=config.batch_sz, shuffle=True)
        return train_dl


if __name__=='__main__':
    import configs.config_tip2022_20230411 as conf_model
    import os

    print('current w_dir ', os.getcwd())
    config = conf_model.generate_config('../configs/train_onet2.0_20231226.yml', dataset_name='zy3')
    print(uti.config_to_str(config))  # print configuration to log file.

    train_dataloader, test_dataloader = cloud_dataloader(config)
    fig_org, axes = plt.subplots(2, 4, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 4))
    # for X, img_ids in train_dataloader:  # train_labels is not use.
    #     for j in range(4):
    #         rgb = torch.permute(X[j,:,:,:], (1,2,0))
    #         rgb = rgb.numpy().astype(np.uint8)
    #         axes[j].imshow(rgb)
    #         axes[j].set_title(img_ids[j])
    #         axes[j].axis('off')
    #     break
    for X, label, img_ids in test_dataloader:  # train_labels is not use.
        for j in range(4):
            rgb = torch.permute(X[j,:,:,:], (1,2,0))
            rgb = rgb.numpy().astype(np.uint8)
            axes[0,j].imshow(rgb) # rgb image has channels at last [h, w, 3] for imshow
            axes[1,j].imshow(label[j, :, :])
            axes[0,j].set_title(img_ids[j], fontsize=8)
            axes[0,j].axis('off')
            axes[1,j].axis('off')
            #fig_org.suptitle('rgb_gray_label')
        plt.pause(0.1)
    plt.show()
    exit(0)