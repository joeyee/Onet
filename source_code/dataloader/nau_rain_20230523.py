'''
Preprocessing the NAU rain dataset based on the 20200819 version.
Created by ZhouYi@Linghai_Dalian on 2023/05/23.
'''


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class NauRainDataset(Dataset):
    def __init__(self, config, img_label_dict):
        #self.img_labels = pd.read_csv(annotations_file)  # pandas read gt label
        #self.img_dir    = config.dir
        #self.echo_names = config.echo_names
        self.use_augmentation = False  # config.use_augmentation
        self.device           = config.device
        self.img_label_dict   = img_label_dict
        self.img_names        = list(self.img_label_dict.keys())
        self.prerpocess() #normalize to [0,1] for each image
    def prerpocess(self):
        for img_name in self.img_names:
            img = self.img_label_dict[img_name]['img']
            #img = img.astype(np.float32)
            img = (img - img.min())/(img.max()-img.min()+np.spacing(1))
            self.img_label_dict[img_name]['img'] = img.to(torch.float32)
        return self.img_label_dict
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_name     = self.img_names[idx]
        img  = self.img_label_dict[image_name]['img']
        label= self.img_label_dict[image_name]['label']
        img  = img.unsqueeze(0)

        return  image_name, img, label

def make_nau_rain_dataloader(config):
    # generate nau_rain dataloader
    # prepare sub_imgs and according labels following the request of config setting.
    data_name = os.path.join(config.dataset_root, config.load_test_file)
    if config.preload:
        img_dict      = torch.load(data_name, map_location=lambda storage, loc: storage)
    else:
        print('Please set preload=True')
        print('Prepare %s first!'%data_name)

    nau_rain_dataset = NauRainDataset(config, img_dict)
    test_dataloader = torch.utils.data.DataLoader(nau_rain_dataset,
                                                   batch_size=config.batch_sz,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   drop_last=False)
    return test_dataloader