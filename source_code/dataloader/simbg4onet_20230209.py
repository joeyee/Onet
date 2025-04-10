'''
Prepare simulated sea clutter background data for Onet training and testing.

sim_bg data, e.g. 'rayleigh_pure_fg_data_2sigma.pt'  is made by 'Rayleigh_bg_Gaussian_EOT_generator_20230208.py', saving at
the location of "configs.dataset_root" 
(train_onet_20250407.yml  dataset_root: "/root/datasets/sim_background/" +   data_file_name: 'rayleigh_2sigma.pt')
'kdist_data.pt'[K-distributed sea clutter with 20 extended targets (Gaussian diffusion)]
'rayleigh_data.pt'[Rayleigh distributed sea clutter with 20 targets.]
each pt file has 7 kind of snrs [0, 2, 4, 5,  6, 8, 10], each snr gets 150 images. so there are
1050 images for one pt file.

Created by ZhouYi@Provence_Dalian @ 20230209, based on 'nauradar4infoseg_20230102.py'
'''

import torch
from   torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import albumentations as A              # do augmentation
import argparse
import utils_20231218 as uti
import os

class simbg4onetDataset(Dataset):
    def __init__(self, config, imgs, labels, snrs, train_or_test="train"):
        self.use_augmentation = config.use_augmentation #default false
        self.device           = config.device
        self.train_or_test    = train_or_test
        self.pixel_transform = A.Compose([
            A.Defocus(p=0.1),
            A.CLAHE(p=0.1), # only support unit8
            A.Equalize(p=0.1),
            A.PixelDropout(p=0.1),
            A.GaussianBlur(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.PixelDropout(p=0.2),
            A.CoarseDropout(p=0.2),
            A.HorizontalFlip(p=0.2)
        ])
        self.imgs   = imgs
        self.labels = labels
        self.snrs   = snrs

        assert(len(self.labels) == len(self.imgs))

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image         = self.imgs[idx, ::]
        snr           = self.snrs[idx]
        label         = self.labels[idx, ::]
        if self.use_augmentation and self.train_or_test == 'train': #augmentation only used for train_samples.
            assert(image.dim()==3)
            image_uint8 =  (image[0,::]*255).to(torch.uint8)   #unit8 for argumentation album input.
            image_aug = self.pixel_transform(image=image_uint8.numpy())['image']  # do albumentations transformation on echo channle only
            image_aug = torch.Tensor(image_aug, dtype='float')
            image_aug = (image_aug - image_aug.min())/(image_aug.max() - image.min() + np.spacing(1))
            image[0,::] = image_aug

        return image, label, snr

def make_simbg4onet_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt",     type=str, default="Adam")
    parser.add_argument("--dataset", type=str, default="simbg")
    parser.add_argument("--dataset_root", type=str, default='/root/datasets/sim_background/') #directory on GPU
    #parser.add_argument("--dataset_root", type=str, default='/home/ubuntu/datasets/sim_background/')
    parser.add_argument("--batch_sz", type=int, default=20)  # batch_sz for dataloader
    #parser.add_argument("--batch_sz", type=int, default=5)  # batch_sz for dataloader

    parser.add_argument("--out_root", type=str, default="./checkpoint/sim_clutter")
    parser.add_argument("--restart", default=False, action="store_true")
    parser.add_argument("--use_augmentation", default=False, action="store_true")
    parser.add_argument("--use_wave_channel", default=False, action="store_true")  # add global wave information in additional wave channel.

    #saving checkpoint frequency
    parser.add_argument("--epoch_nums", type=int, default=1000)
    # parser.add_argument("--save_freq", type=int, default=200)
    # parser.add_argument("--gt_k",      type=int, default=3)

    # input size for the infoseg. It defines the sub_rect size of the radar frame.
    parser.add_argument("--input_sz",   type=int, default=224)

    # parser.add_argument("--cfar_kval", type=float, default=1.0)

    config = parser.parse_args()  # get command line parameters

    #echo_names is the source raw radar data.
    #config.echo_names = [os.path.join(config.dataset_root, 'rayleigh_pure_fg_data_2sigma.pt')]
    config.preload    = True #preload the rayleigh_pure_fg_data_2sigma.pt
    if torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'
    return config

def make_simbg_dataloader(simbg_config):
    '''
    Prepare train and test dataloader for training and testing
    :param nau_config:
    :return:
    '''
    # prepare sub_imgs and according labels following the request of config setting.
    data_name = os.path.join(simbg_config.dataset_root, simbg_config.data_file_name)
    if simbg_config.preload:
        data      = torch.load(data_name, map_location=lambda storage, loc: storage)
        imgs      = data['rayleigh_imgs']
        imgs      = uti.tensor_normal_per_frame(imgs)
        labels    = data['rayleigh_labels']
        snrs      = torch.tensor(data['psnr'])

    else:
        print('Please set preload=True')
        print('Prepare %s first!'%data_name)
        exit(-1)

    # Divide the samples into train_samples[90%] and test_samples[10%]
    nsamples = imgs.shape[0]
    ntrain_nums = int(nsamples * 0.9)
    ntest_nums = nsamples - ntrain_nums
    ids = np.arange(nsamples)
    np.random.shuffle(ids)
    train_idx = ids[:ntrain_nums]
    test_idx = ids[ntrain_nums:ntrain_nums + ntest_nums]
    train_imgs = imgs[train_idx, ::]
    train_labels = labels[train_idx, ::]
    train_snrs = snrs[train_idx]

    test_imgs = imgs[test_idx, ::]
    test_labels = labels[test_idx, ::]
    test_snrs = snrs[test_idx]
    # prepare dataset based on pytorch format
    simbg_train_dataset = simbg4onetDataset(simbg_config, train_imgs, train_labels,train_snrs,
                                                train_or_test='train')  # train without labels.
    simbg_test_dataset  = simbg4onetDataset(simbg_config, test_imgs, test_labels, test_snrs,
                                                train_or_test='test')

    # generate the train and test loader
    train_dataloader = torch.utils.data.DataLoader(simbg_train_dataset,
                                                   batch_size=simbg_config.batch_sz,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(simbg_test_dataset,
                                                  batch_size=simbg_config.batch_sz,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=False)
    return train_dataloader, test_dataloader

def make_dataloader_via_snr(simbg_config, snr, bshuffle=True):
    '''
    Prepare train and test dataloader for training and testing based on snr
    :param nau_config:
    :param snr: range from 0 to 10
    :return:
    '''
    # prepare sub_imgs and according labels following the request of config setting.
    data_name = os.path.join(simbg_config.dataset_root, simbg_config.data_file_name)
    if simbg_config.preload:
        data      = torch.load(data_name, map_location=lambda storage, loc: storage)
        #print(data.keys())
        imgs      = data['rayleigh_imgs']
        imgs      = uti.tensor_normal_per_frame(imgs)
        labels    = data['rayleigh_labels']
        snrs      = torch.tensor(data['psnr'])

    else:
        print('Please set preload=True')
        print('Prepare %s first!'%data_name)

    index = torch.where(snrs==snr)[0]
    if (len(index)==0):
        print('snr=%d is not in the dataset'%snr)
        return None
    assert(torch.all(snrs[index]==snr)) #assert all snrs are equal to snr
    sel_imgs   = imgs[index]
    sel_labels = labels[index]
    sel_snrs   = snrs[index]
    # prepare dataset based on pytorch format
    simbg_train_dataset = simbg4onetDataset(simbg_config, sel_imgs, sel_labels,sel_snrs, train_or_test='train')  # aug for trains only

    # generate the selected_data_loader
    dataloader = torch.utils.data.DataLoader(simbg_train_dataset,
                                                   batch_size=simbg_config.batch_sz,
                                                   shuffle=bshuffle, #shuffle  default is True
                                                   num_workers=0,
                                                   drop_last=False)

    return dataloader

def make_datasets_via_snr(simbg_config, snr):
    # prepare sub_imgs and according labels following the request of config setting.
    data_name = os.path.join(simbg_config.dataset_root, simbg_config.data_file_name)
    if simbg_config.preload:
        data = torch.load(data_name, map_location=lambda storage, loc: storage)
        # print(data.keys())
        imgs = data['rayleigh_imgs']
        imgs = uti.tensor_normal_per_frame(imgs)
        labels = data['rayleigh_labels']
        snrs = torch.tensor(data['psnr'])

    else:
        print('Please set preload=True')
        print('Prepare %s first!' % data_name)

    index = torch.where(snrs == snr)[0]
    if (len(index) == 0):
        print('snr=%d is not in the dataset' % snr)
        return None
    assert (torch.all(snrs[index] == snr))  # assert all snrs are equal to snr
    sel_imgs = imgs[index]
    sel_labels = labels[index]
    sel_snrs = snrs[index]

    # Divide the samples into train_samples[90%] and test_samples[10%]
    nsamples = sel_imgs.shape[0]
    ntrain_nums = int(nsamples * 0.9)
    ntest_nums = nsamples - ntrain_nums
    ids = np.arange(nsamples)
    np.random.shuffle(ids)
    train_idx = ids[:ntrain_nums]
    test_idx = ids[ntrain_nums:ntrain_nums + ntest_nums]
    train_imgs = sel_imgs[train_idx, ::]
    train_labels = sel_labels[train_idx, ::]
    train_snrs = sel_snrs[train_idx]

    test_imgs = sel_imgs[test_idx, ::]
    test_labels = sel_labels[test_idx, ::]
    test_snrs = sel_snrs[test_idx]

    # prepare dataset based on pytorch format
    train_dataset_per_snr = simbg4onetDataset(simbg_config, train_imgs, train_labels, train_snrs,
                                            train_or_test='train')  # train without labels.
    test_dataset_per_snr  = simbg4onetDataset(simbg_config, test_imgs, test_labels, test_snrs,
                                            train_or_test='test')
    return train_dataset_per_snr, test_dataset_per_snr

def make_dataloader_eq_dist_in_snr_range(simbg_config, low_snr=1, high_snr=1):
    '''
    Make equalized distribution in the SNR Range.
    :param simbg_config:
    :param low_snr:
    :param high_snr:
    :return:
    '''
    # prepare sub_imgs and according labels following the request of config setting.
    data_name = os.path.join(simbg_config.dataset_root, simbg_config.data_file_name)
    if simbg_config.preload:
        data = torch.load(data_name, map_location=lambda storage, loc: storage)
        # print(data.keys())
        imgs = data['rayleigh_imgs']
        imgs = uti.tensor_normal_per_frame(imgs)
        labels = data['rayleigh_labels']
        snrs = torch.tensor(data['psnr'])
    else:
        print('Please set preload=True')
        print('Prepare %s first!' % data_name)
    # assert(snr==5)
    assert (high_snr >= low_snr)
    snr = low_snr
    train_ds_list = []
    test_ds_list = []
    while snr <= high_snr:
        train_dataset_per_snr, test_dataset_per_snr = make_datasets_via_snr(simbg_config, snr)
        train_ds_list.append(train_dataset_per_snr)
        test_ds_list.append(test_dataset_per_snr)
        snr +=1
    train_dataset_snrs = ConcatDataset(train_ds_list)
    test_dataset_snrs  = ConcatDataset(test_ds_list)

    # generate the selected_data_loader
    # generate the selected_data_loader
    train_dataloader_snrs = torch.utils.data.DataLoader(train_dataset_snrs,
                                                   batch_size=simbg_config.batch_sz,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   drop_last=False)

    test_dataloader_snrs = torch.utils.data.DataLoader(test_dataset_snrs,
                                                  batch_size=simbg_config.batch_sz,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=False)
    return train_dataloader_snrs, test_dataloader_snrs

def make_dataloader_snr_range(simbg_config, low_snr=1, high_snr=1):
    '''
    Prepare train and test dataloader for training and testing based on snr
    :param nau_config:
    :param snr: range from 0 to 10
    :param mode: 'equal', 'high' or  'low'
    :return:
    '''
    # prepare sub_imgs and according labels following the request of config setting.
    data_name = os.path.join(simbg_config.dataset_root, simbg_config.data_file_name)
    if simbg_config.preload:
        data      = torch.load(data_name, map_location=lambda storage, loc: storage)
        #print(data.keys())
        imgs      = data['rayleigh_imgs']
        imgs      = uti.tensor_normal_per_frame(imgs)
        labels    = data['rayleigh_labels']
        snrs      = torch.tensor(data['psnr'])
    else:
        print('Please set preload=True')
        print('Prepare %s first!' % data_name)
    #assert(snr==5)
    assert(high_snr>=low_snr)
    mask = (snrs<=high_snr) & (snrs>=low_snr)
    index = torch.where(mask)[0]
    assert (torch.all(snrs[index] <= high_snr))  #
    assert (torch.all(snrs[index] >= low_snr))   #

    if (len(index)==0):
        print('snr %d-%d is not in the range %d-%d'
              %(torch.min(snrs).item(), torch.max(snrs).item(), low_snr, high_snr))
        return None
    # assert(torch.all(snrs[index]==snr)) #assert all snrs are equal to snr
    sel_imgs   = imgs[index]
    sel_labels = labels[index]
    sel_snrs   = snrs[index]

    # Divide the samples into train_samples[90%] and test_samples[10%]
    nsamples = sel_imgs.shape[0]
    ntrain_nums = int(nsamples * 0.9)
    ntest_nums = nsamples - ntrain_nums
    ids = np.arange(nsamples)
    np.random.shuffle(ids)
    train_idx = ids[:ntrain_nums]
    test_idx = ids[ntrain_nums:ntrain_nums + ntest_nums]
    train_imgs = sel_imgs[train_idx, ::]
    train_labels = sel_labels[train_idx, ::]
    train_snrs = sel_snrs[train_idx]

    test_imgs = sel_imgs[test_idx, ::]
    test_labels = sel_labels[test_idx, ::]
    test_snrs = sel_snrs[test_idx]
    # prepare dataset based on pytorch format
    simbg_train_dataset = simbg4onetDataset(simbg_config, train_imgs, train_labels, train_snrs,
                                            train_or_test='train')  # train without labels.
    simbg_test_dataset = simbg4onetDataset(simbg_config, test_imgs, test_labels, test_snrs,
                                           train_or_test='test')

    # generate the selected_data_loader
    train_dataloader = torch.utils.data.DataLoader(simbg_train_dataset,
                                                   batch_size=simbg_config.batch_sz,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(simbg_test_dataset,
                                                  batch_size=simbg_config.batch_sz,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=False)
    return train_dataloader, test_dataloader
if __name__ == '__main__':
    #exit(0)
    torch.manual_seed(1981)
    np.random.seed(1981)
    torch.set_default_dtype(torch.float32)

    print('simbg4onet_20230209: ', os.getpid())
    simbg_config  = make_simbg4onet_config()  # parsing options
    simbg_config.data_file_name = 'rayleigh_pure_fg_data_2sigma.pt'

    simbg_config.preload  = True #directly load train_test data.
    simbg_config.batch_sz = 20
    print(uti.config_to_str(simbg_config))

    #train_dataloader, test_dataloader = make_simbg_dataloader(simbg_config)
    train_dataloader, test_dataloader = make_dataloader_eq_dist_in_snr_range(simbg_config, low_snr=0, high_snr=2)
    #print(view_source_data(nau_config))
    fig, axs = plt.subplots(2,5,gridspec_kw={'wspace': 0.01, 'hspace': 0.01}, figsize=(8, 3.2), sharex=True, sharey=True)
    fig.suptitle('First 5 images of train_loader')
    for imgs, labels, snrs in iter(train_dataloader): # get one batch_sz of imgs and labels for viewing.
        #print(img_dict)
        n = min(5, imgs.shape[0]) # at most show 5 pictures in one row.
        for i in range(n):
            axs[0,i].imshow(imgs[i, 0, ::].numpy())
            # if imgs.shape[1]>1: #use_wave_channel==True
            #     axs[1,i].imshow(imgs[i, 1, ::].numpy()) # wave information
            axs[1,i].imshow(labels[i, ::].numpy())
            axs[0,i].set_title('snr: '+str(snrs[i].item()))
            axs[1, i].set_xticks([])
            axs[0, i].set_yticks([])
        plt.show()
        exit(0)

