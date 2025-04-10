'''
Test the performance of the pre-processing on the ZY3 testset.
The pre-processing includes raw_rgb, [histeq_rgb, contrast_enhance], [haze_remove, haze_enhance].
or using haze_remove and haze_enhance after histeq_rgb and contrast_enhance.

Using contrast_enhance on thin_cloud.
Using haze_remove on snow_foreground.

Choose the one with the best miou performance.
Created by ZhouYi@Linghai_Dalian on 20240321

Modified by ZhouYi@Linghai_Dalian on 20240607: Adapt to Onet's model after 202406. such as Onet_vanilla_20240606

'''


from datetime import datetime
import torch
import torch.nn as nn
import utils_20231218 as uti
import configs.config_tip2022_20230411           as conf_model
import dataloader.zy3_cloud_thumbnailv5_20240304 as cloud_model_zy3
import uti_zy3_test_20240123 as uti_zy3_test
#import Onet_vanilla_outc_20240305                as onet_vanilla_outc
import Onet_vanilla_20240606                     as onet_vanilla_model
import select_trainset_for_correct_clouds_20240307 as select_trainset
import torch.nn.functional as F
import copy
import logging
import os
import albumentations as A
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import haze_remove_20240313   as haze_remove_model
import utils_20231218 as uti
import pandas as pd
from PIL import Image, ImageEnhance



torch.manual_seed(1981)
np.random.seed(1981)
torch.set_default_dtype(torch.float32)

def contrast_enhance_random(image):
    '''
    This function is used to enhance the contrast of the image randomly.
    :param img: PIL image
    :return:    PIL image
    '''
    image = np.array(image)
    Atransform = A.Compose([
        A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(0.04, 0.38), contrast_limit=(-0.19, 0.35), brightness_by_max=True)
    ])
    image_trans = Atransform(image=image)['image']
    image_trans = Image.fromarray(image_trans)
    return image_trans

def contrast_enhance(img, factor=0.5):
    '''
    This function is used to enhance the contrast of the image determinately via PIL's enhance.
    Following the information on
    https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/ImageEnhance.html
    :param img: PIL image
    :return:    PIL image
    '''
    contrast_enhancer = ImageEnhance.Contrast(img)
    img      = contrast_enhancer.enhance(factor)
    # bright_enhancer = ImageEnhance.Brightness(img)
    # img = bright_enhancer.enhance(factor)
    return img
def get_image_depth_via_haze_remove(I):
    '''
    Get the image's depth according to He Kaiming's haze removing paper(cvpr2009).
    t = exp(-beta*depth) , t means the medium transmission, beta is the empirical scale parameters.
    depth = -log(t)/beta
    :param I:
    :return:
    '''
    dark = haze_remove_model.DarkChannel(I, sz=3) # dark channel
    A = haze_remove_model.AtmLight(I, dark)         # gloabl atmospheric light
    te = haze_remove_model.TransmissionEstimate(I, A, sz=3) # transmission estimate
    t = haze_remove_model.TransmissionRefine(I, te, radius=3, eps=0.0001) # transmission refine with guided image I
    #J = haze_remove_model.Recover(I, t, A, 0.1) # dark object's radiance
    #J = haze_remove_model.Recover(I, t, A, 0.2) # dark object's radiance
    J = haze_remove_model.Recover(I, t, A, 0.3)  # dark object's radiance
    K = A.max()*(1-t) #clouds' radian
    K = K.astype(np.float32)
    return J, K


def make_thrumnail_image(file, pre_option):
    '''pre-processing options: raw_rgb, histeq_rgb, contrast_enhance, haze_remove, haze_enhance'''
    if 'pre'  in file:
        pid = file.split("_")[-2]
    else:
        pid = file.split("_")[-1].split(".")[0]

    img = Image.open(file)
    # preprocessing is done in the bigger image, such as contrast enhance, histogram equalization
    if img.mode == 'L':
        #img = np.stack((img, img, img), axis=2)
        img = img.convert('RGB') #convert gray image to rgb
    else:
        assert img.mode == 'RGB', 'the image mode is not RGB'
    img_trans = transforms.Resize((300))(img) # resize to 300x300 first
    #img.thumbnail((300,300)) # or #img.resize((300,300), img.ANTIALIAS) #PIL resize is better in quality
    img = transforms.CenterCrop(224)(img_trans) #current still ipl image

    #totally 8 pre-processing options, no single haze_remove.
    assert pre_option in ['raw_rgb', #no pre-processing
                          'histeq_rgb', 'contrast_enhance', 'haze_enhance', 'haze_remove', #single pre-processing
                          'histeq_haze_enhance', 'histeq_haze_remove', #combined pre-processing
                          'contrast_enhance_haze_enhance', 'contrast_enhance_haze_remove'] #combined pre-processing
    # omit 'haze_remove'
    if pre_option == 'raw_rgb':
        pass # not do any preprocessing
    if pre_option == 'histeq_rgb':
        img = transforms.functional.equalize(img)  # histogram equalization on PIL image
    if pre_option == 'contrast_enhance':
        img = contrast_enhance(img)                # contrast enhance on PIL image
    if pre_option == 'haze_remove':
        # if pid == '1706195991':
        #     print('processing image: ', pid)
        I = np.array(img).astype('float32') / 255.
        J, K = get_image_depth_via_haze_remove(I)  # haze remove and enhance on PIL image
        J = np.clip(J, 0, 1.)
        img = Image.fromarray((J*255).astype('uint8'))

    if pre_option == 'haze_enhance':
        I = np.array(img).astype('float32') / 255.  # normalize the image to ndarray with range [0,1.]
        J, K = get_image_depth_via_haze_remove(I)  # haze remove and enhance on float ndarray
        I  = I + np.stack([ K, K,  K], axis=2)
        I  = np.clip(I, 0, 1.)
        img = Image.fromarray((I*255).astype('uint8'))

    if pre_option == 'histeq_haze_enhance': #histogram equalization and haze enhance combined
        img = transforms.functional.equalize(img)
        I = np.array(img).astype('float32') / 255.  # normalize the image to ndarray with range [0,1.]
        J, K = get_image_depth_via_haze_remove(I)  # haze remove and enhance on float ndarray
        I = I + np.stack([1.7 * K, 1.7 * K, 1.7 * K], axis=2)
        I = np.clip(I, 0, 1.)
        img = Image.fromarray((I * 255).astype('uint8')) # convert ndarray to PIL image

    if pre_option == 'histeq_haze_remove':
        img = transforms.functional.equalize(img)
        I = np.array(img).astype('float32') / 255.
        J, K = get_image_depth_via_haze_remove(I)  # haze remove and enhance on PIL image
        J = np.clip(J, 0, 1.)
        img = Image.fromarray((J * 255).astype('uint8'))

    if pre_option == 'contrast_enhance_haze_enhance':
        img = contrast_enhance(img)
        I = np.array(img).astype('float32') / 255.
        J, K = get_image_depth_via_haze_remove(I)
        I = I + np.stack([1.7 * K, 1.7 * K, 1.7 * K], axis=2)
        I = np.clip(I, 0, 1.)
        img = Image.fromarray((I * 255).astype('uint8'))

    if pre_option == 'contrast_enhance_haze_remove':
        img = contrast_enhance(img)
        I = np.array(img).astype('float32') / 255.
        J, K = get_image_depth_via_haze_remove(I)
        J = np.clip(J, 0, 1.)
        img = Image.fromarray((J * 255).astype('uint8'))

    img_thumbnail_224 = transforms.ToTensor()(img) #conver img to tensor

    # img_trans         = transforms.Resize((300))(img)
    # img_thumbnail_224 = transforms.CenterCrop(224)(img_trans)
    # img_thumbnail_224 = transforms.ToTensor()(img_thumbnail_224)


    assert img_thumbnail_224.dim()==3, 'the thumbnail size is not 224x224x3'
    if img_thumbnail_224.shape[0] ==1: #gray source image
        img_thumbnail_224 = torch.vstack((img_thumbnail_224, img_thumbnail_224, img_thumbnail_224))
    return img_thumbnail_224, pid

def make_thumnail_mask(file):
    '''
    Resize mask and central crop to 224x224, and convert to tensor.
    All 1 marks for tid with '1706158599'.
    :param file:
    :return:
    '''
    if 'pre' in file:
        pid = file.split("_")[-2]
    else:
        pid = file.split("_")[-1].split(".")[0]
    img = Image.open(file)

    #img.thumbnail((300, 300))  # or #img.resize((300,300), img.ANTIALIAS) #PIL resize is better in quality
    img_trans = transforms.Resize((300))(img)

    img_thumbnail_224 = transforms.CenterCrop(224)(img_trans)
    img_thumbnail_224 = transforms.ToTensor()(img_thumbnail_224)

    img_thumbnail_224 = img_thumbnail_224[0, ...]
    img_thumbnail_224 = torch.where(img_thumbnail_224 > 0.5, 1.0, 0)
    if (pid == '1706158599'):
        # img_thumbnail_224 = torch.ones
        print('non-cloud sums in 1706158599 ', torch.sum(img_thumbnail_224 == 0))
        img_thumbnail_224 = torch.ones_like(img_thumbnail_224)
        print('mark it as all 1!!!')
    return img_thumbnail_224, pid

def test_onet_on_single_image(config, onet, img, label):
    '''
    Test the onet on single image and mask.
    :param onet:
    :param img:
    :param label:
    :return: acc and miou
    '''
    onet.eval()
    with torch.no_grad():
        X = img.unsqueeze(0).to(config.device)
        label = label.to(config.device)

        Lt, Vt, Ld, Vd, S = onet(X)  # send rgb to onet for convnext backbone
        # pred_t = Vt.squeeze(dim=1)
        # pred_d = Vd.squeeze(dim=1)
        pred_labels = onet.predict_label(S)
        Y = uti.reorder_segmentation(pred_labels, label)

        acc  = uti._acc(pred_labels[0,::], label, 2)
        miou = uti._miou(pred_labels[0,::], label, 2)
        # acc1, miou1 = uti.evaluate_segmentation(pred_label, label, gt_k=2) # cause confusion in snow foreground
        # if acc1!= acc:
        #     print('acc is not equal, pred_acc %.4f, modified_acc %.4f' % (acc, acc1))
    return acc, miou

def get_cloud_snr(img, mask):
    '''
    get the cloud snr and scr (single to clutter ratio) from the mask image
    :param img: the image with the cloud
    :param mask: the mask image with the cloud
    :return:
    '''
    mask = torch.stack((mask, mask, mask), dim=0)
    assert img.shape==mask.shape, 'the image and mask shape is not the same'
    cloud     = img[mask>0]
    noncloud  = img[mask==0]
    assert  mask.sum()>0, 'the mask is empty' # must get cloud mask.
    cloud_snr = cloud.mean()/( noncloud.std()  + np.spacing(1))    # avoid the zero division
    cloud_snr = 20*torch.log10(cloud_snr) # in dB.

    cloud_scr = cloud.mean()/(noncloud.mean()+np.spacing(1)) # signel to clutter ratio
    cloud_scr = 20*torch.log10(cloud_scr)
    #noncloud_snr = noncloud.mean()/noncloud.std()
    #print('zy3_test_%s, cloud_snr, %.2f, cloud_scr, %.2f'%(pid, cloud_snr, cloud_scr))
    return cloud_snr.item(), cloud_scr.item()

def classified_preprocess(config, onet, zy3_divided_test_ids_dict):
    pre_options = ['raw_rgb', 'histeq_rgb', 'contrast_enhance', 'haze_enhance', 'haze_remove',  # single pre-processing
                   'histeq_haze_enhance', 'histeq_haze_remove', 'contrast_enhance_haze_remove',
                   # combined pre-processing
                   'contrast_enhance_haze_enhance']  # strong enhance only for 1706158599
    # 'contrast_enhance_haze_enhance'] #too strong
    path_src_list = [r'/Users/yizhou/Datasets/zy-3_thumbnail_dataset/test-imgs']
    path_gt_list = [r'/Users/yizhou/Datasets/zy-3_thumbnail_dataset/test_label_50_255']
    path_src_list = [r'/root/datasets/zy-3_thumbnail_dataset/test-imgs']
    path_gt_list = [r'/root/datasets/zy-3_thumbnail_dataset/test_label_50_255']
    path_src_list = [r'/root/datasets/thumbnail224/test-imgs']
    path_gt_list = [r'/root/datasets/thumbnail224/test_label_50_255']
    out_root_dict = r'/root/datasets/thumbnail224/zy3_thumbnail224_test_label_dict50_bestACC_preprocess.pt'


    for path_src, path_gt in zip(path_src_list, path_gt_list):
        src_files = glob(os.path.join(path_src, '*.jpg'))
        src_files.extend(glob(os.path.join(path_src, '*.JPG')))
        gt_files = glob(os.path.join(path_gt, '*.png'))
        src_files.sort()
        gt_files.sort()
        image_thumbnail_test_dict = {}
        res_dict_list = []  # list for sort miou
        # fig, axs = plt.subplots(2, 1, figsize=(4, 8))  # 512x512 resolution
        print('test_sample_nums:', len(src_files))
        for srcfile, gt_file in tqdm(zip(src_files, gt_files)):
            best_miou = 0
            best_acc = 0
            img_org, _ = make_thrumnail_image(srcfile, 'raw_rgb')
            gt_org, _ = make_thumnail_mask(gt_file)
            org_snr, org_scr = get_cloud_snr(img_org, gt_org)
            gt_thumb, gid  = make_thumnail_mask(gt_file)
            img_ids = 'zy3_test_' + gid

            classified_type = ''
            if img_ids in zy3_divided_test_ids_dict['normal_cloud']:
                classified_type = 'normal_cloud'
                pre_opt = 'haze_enhance'
            elif img_ids in zy3_divided_test_ids_dict['thin_cloud']:
                classified_type = 'thin_cloud'
                pre_opt = 'haze_enhance'
            elif img_ids in zy3_divided_test_ids_dict['snow_cloud']:
                classified_type = 'snow_cloud'
                pre_opt = 'contrast_enhance_haze_remove'
            img_thumb, mid = make_thrumnail_image(srcfile, pre_opt)

            assert (mid == gid)
            # if pre_opt == 'contrast_enhance_haze_enhance' and mid != '1706158599':  # skip the strong enhance for other images.
            #     continue
            acc, miou = test_onet_on_single_image(config, onet, img_thumb, gt_thumb)
            image_thumbnail_test_dict['zy3_test_' + mid] = {'true_color': img_thumb, 'mask': gt_thumb}
            image_thumbnail_test_dict['zy3_test_' + mid]['miou'] = miou
            image_thumbnail_test_dict['zy3_test_' + mid]['acc'] = acc
            image_thumbnail_test_dict['zy3_test_' + mid]['opt'] = pre_opt
            pre_snr, pre_scr = get_cloud_snr(img_thumb, gt_thumb)
            image_thumbnail_test_dict['zy3_test_' + mid]['org_snr'] = org_snr
            image_thumbnail_test_dict['zy3_test_' + mid]['org_scr'] = org_scr
            image_thumbnail_test_dict['zy3_test_' + mid]['pre_snr'] = pre_snr
            image_thumbnail_test_dict['zy3_test_' + mid]['pre_scr'] = pre_scr

        for key in image_thumbnail_test_dict.keys():
            res_dict_list.append({'img_id': key, 'miou': image_thumbnail_test_dict[key]['miou'],
                                  'acc': image_thumbnail_test_dict[key]['acc'],
                                  'opt': image_thumbnail_test_dict[key]['opt'],
                                  'org_snr': image_thumbnail_test_dict[key]['org_snr'],
                                  'org_scr': image_thumbnail_test_dict[key]['org_scr'],
                                  'pre_snr': image_thumbnail_test_dict[key]['pre_snr'],
                                  'pre_scr': image_thumbnail_test_dict[key]['pre_scr']})

        print('saving the thumbnails with the best pre-processing option to the dict file: ', out_root_dict)
        torch.save(image_thumbnail_test_dict, out_root_dict)
        res_dict_list.sort(key=lambda x: x['miou'], reverse=True)
        mean_acc = 0
        mean_miou = 0
        mean_org_snr = 0
        mean_pre_snr = 0
        nsamples = len(res_dict_list)
        for img_dict in res_dict_list:
            mean_acc += img_dict['acc']
            mean_miou += img_dict['miou']
            mean_org_snr += img_dict['org_snr']
            mean_pre_snr += img_dict['pre_snr']
            classified_type = ''
            if img_dict['img_id'] in zy3_divided_test_ids_dict['normal_cloud']:
                classified_type = 'normal_cloud'
            elif img_dict['img_id'] in zy3_divided_test_ids_dict['thin_cloud']:
                classified_type = 'thin_cloud'
            elif img_dict['img_id'] in zy3_divided_test_ids_dict['snow_cloud']:
                classified_type = 'snow_cloud'
            print('%s,\t input,%10s,acc,%.4f,miou,%.4f, classified type, %s' %
                  (img_dict['img_id'], img_dict['opt'], img_dict['acc'], img_dict['miou'], classified_type))
        print('acc %.4f, miou %.4f after pre-processing ' % (mean_acc / nsamples, mean_miou / nsamples))
        # write res_dict_list to excel file
        df = pd.DataFrame(res_dict_list)
        # df = df.drop(columns=['rgb_file', 'label_file', 'pred_file'])
        df.to_excel(os.path.join(config.out_root, 'zy3_testset50_best_preprocess202406.xlsx'), index=False)
        print('saving the best pre-processing option to the excel file: ', os.path.join(config.out_root, 'zy3_testset50_classified_preprocess202406.xlsx'))

def choose_test_preprocess(config, onet):
    '''
    Load the testfiles to thumnail images and masks upon the pre-processing options.
    Choose the best pre-processing option according to the best miou performance.
    Save the best pre-processing option to the excel file.
    :return: dict of thumnail images and masks in float tensor.
    '''
    pre_options  = ['raw_rgb', 'histeq_rgb', 'contrast_enhance', 'haze_enhance', 'haze_remove', #single pre-processing
                          'histeq_haze_enhance', 'histeq_haze_remove','contrast_enhance_haze_remove',#combined pre-processing
                    'contrast_enhance_haze_enhance'] #strong enhance only for 1706158599
    #'contrast_enhance_haze_enhance'] #too strong
    path_src_list = [r'/Users/yizhou/Datasets/zy-3_thumbnail_dataset/test-imgs']
    path_gt_list = [r'/Users/yizhou/Datasets/zy-3_thumbnail_dataset/test_label_50_255']
    path_src_list = [r'/root/datasets/zy-3_thumbnail_dataset/test-imgs']
    path_gt_list = [r'/root/datasets/zy-3_thumbnail_dataset/test_label_50_255']
    path_src_list = [r'/root/datasets/thumbnail224/test-imgs']
    path_gt_list = [r'/root/datasets/thumbnail224/test_label_50_255']
    out_root_dict = r'/root/datasets/thumbnail224/zy3_thumbnail224_test_label_dict50_bestACC_preprocess.pt'

    for path_src, path_gt in zip(path_src_list, path_gt_list):
        src_files = glob(os.path.join(path_src, '*.jpg'))
        src_files.extend(glob(os.path.join(path_src, '*.JPG')))
        gt_files = glob(os.path.join(path_gt, '*.png'))
        src_files.sort()
        gt_files.sort()
        image_thumbnail_test_dict = {}
        res_dict_list = [] #list for sort miou
        # fig, axs = plt.subplots(2, 1, figsize=(4, 8))  # 512x512 resolution
        print('test_sample_nums:', len(src_files))
        for srcfile, gt_file in tqdm(zip(src_files, gt_files)):
            best_miou = 0
            best_acc  = 0
            img_org, _ = make_thrumnail_image(srcfile, 'raw_rgb')
            gt_org, _  = make_thumnail_mask(gt_file)
            org_snr, org_scr = get_cloud_snr(img_org, gt_org)

            for pre_opt in pre_options:
                img_thumb, mid = make_thrumnail_image(srcfile, pre_opt)
                gt_thumb, gid  = make_thumnail_mask(gt_file)
                assert (mid == gid)
                img_ids = 'zy3_test_' + gid
                classified_type = ''
                if img_ids in zy3_divided_test_ids_dict['normal_cloud']:
                    classified_type = 'normal_cloud'
                    #pre_opt = 'raw_rgb'
                elif img_ids in zy3_divided_test_ids_dict['thin_cloud']:
                    classified_type = 'thin_cloud'
                    #pre_opt = 'haze_enhance'
                elif img_ids in zy3_divided_test_ids_dict['snow_cloud']:
                    classified_type = 'snow_cloud'
                    #pre_opt = 'haze_remove'

                if pre_opt == 'contrast_enhance_haze_enhance' and mid != '1706158599': #skip the strong enhance for other images.
                    continue
                # if mid == '17042767336420149' or mid=='17042505436415835':
                #     print('processing snow foreground image: ', mid)
                acc, miou = test_onet_on_single_image(config, onet, img_thumb, gt_thumb)
                if miou > best_miou: # choose the best miou across all the pre-processing options.
                #if acc > best_acc:
                    best_miou = miou
                    #best_acc = acc
                    image_thumbnail_test_dict['zy3_test_' + mid] = {'true_color': img_thumb, 'mask': gt_thumb}
                    image_thumbnail_test_dict['zy3_test_' + mid]['miou']= miou
                    image_thumbnail_test_dict['zy3_test_' + mid]['acc'] = acc
                    image_thumbnail_test_dict['zy3_test_' + mid]['opt'] = pre_opt
                    pre_snr, pre_scr = get_cloud_snr(img_thumb, gt_thumb)
                    image_thumbnail_test_dict['zy3_test_' + mid]['org_snr'] = org_snr
                    image_thumbnail_test_dict['zy3_test_' + mid]['org_scr'] = org_scr
                    image_thumbnail_test_dict['zy3_test_' + mid]['pre_snr'] = pre_snr
                    image_thumbnail_test_dict['zy3_test_' + mid]['pre_scr'] = pre_scr
                    image_thumbnail_test_dict['zy3_test_' + mid]['classified_type'] = classified_type
                if mid == '1706158599':
                    print('processing thin-cloud image: ', mid, pre_opt, ' acc ', acc, ' best_acc ', best_acc)


        for key in image_thumbnail_test_dict.keys():
            res_dict_list.append({'img_id': key, 'miou': image_thumbnail_test_dict[key]['miou'],
                                  'acc': image_thumbnail_test_dict[key]['acc'],
                                  'opt': image_thumbnail_test_dict[key]['opt'],
                                  'org_snr': image_thumbnail_test_dict[key]['org_snr'],
                                  'org_scr': image_thumbnail_test_dict[key]['org_scr'],
                                  'pre_snr': image_thumbnail_test_dict[key]['pre_snr'],
                                  'pre_scr': image_thumbnail_test_dict[key]['pre_scr'],
                                  'classified_type': image_thumbnail_test_dict[key]['classified_type']})


        print('saving the thumbnails with the best pre-processing option to the dict file: ', out_root_dict)
        torch.save(image_thumbnail_test_dict, out_root_dict)
        res_dict_list.sort(key=lambda x: x['miou'], reverse=True)
        mean_acc = 0
        mean_miou = 0
        mean_org_snr = 0
        mean_pre_snr = 0
        nsamples = len(res_dict_list)
        for img_dict in res_dict_list:
            mean_acc += img_dict['acc']
            mean_miou += img_dict['miou']
            mean_org_snr += img_dict['org_snr']
            mean_pre_snr += img_dict['pre_snr']
            classified_type = ''
            if img_dict['img_id'] in zy3_divided_test_ids_dict['normal_cloud']:
                classified_type = 'normal_cloud'
            elif img_dict['img_id'] in zy3_divided_test_ids_dict['thin_cloud']:
                classified_type = 'thin_cloud'
            elif img_dict['img_id'] in zy3_divided_test_ids_dict['snow_cloud']:
                classified_type = 'snow_cloud'
            print('%s,\t input,%10s,acc,%.4f,miou,%.4f, classified type, %s' %
                  (img_dict['img_id'], img_dict['opt'], img_dict['acc'], img_dict['miou'], classified_type))
        print('acc %.4f, miou %.4f after pre-processing ' % (mean_acc / nsamples, mean_miou / nsamples))
        # write res_dict_list to excel file
        df = pd.DataFrame(res_dict_list)
        # df = df.drop(columns=['rgb_file', 'label_file', 'pred_file'])
        df.to_excel(os.path.join(config.out_root, 'zy3_testset50_best_preprocess202406.xlsx'), index=False)
        print('saving the best pre-processing option to the excel file: ', os.path.join(config.out_root, 'zy3_testset50_best_preprocess202406.xlsx'))


def prepare_train_thumbnails():
    '''
    using the source train images with size 1kx1k to generate the thumbnails with size 224x224x3
    via plt.figure() and plt.savefig(), and centercrop(224) to get the 224x224x3 thumbnails.
    :return:
    '''
    data_root = r'/Users/yizhou/Datasets/zy-3_thumbnail_dataset/train*/'
    data_root = r'/root/datasets/zy-3_thumbnail_dataset/train*/'

    out_root_dict    = r'/root/datasets/thumbnail224/zy3_thumbnail224_train_dict250_v2.pt' #tensor resize, histeq, centercrop
    out_root_dict = r'/root/datasets/thumbnail224/zy3_thumbnail224_train_dict250_v3.pt' # iplimage resize, histeq in ndarray
    train_dir = glob(data_root)
    train_dir.sort()
    train_sample_nums = 0
    src_files = []
    #fig, axs = plt.subplots(1, 2, figsize=(10,5))  # 512x512 resolution
    for path_src in train_dir:
        jpg_files = glob(os.path.join(path_src, '*.jpg'))
        JPG_files = glob(os.path.join(path_src, '*.JPG'))
        src_files.extend(jpg_files)
        src_files.extend(JPG_files)
    src_files.sort()
    train_sample_nums += len(src_files)
    print('train_sample_nums:', train_sample_nums)
    image_thumbnails_train_dict = {}
    for srcfile in tqdm(src_files):
        img_thumb, pid = make_thrumnail_image(srcfile, 'raw_rgb')
        image_thumbnails_train_dict['zy3_train_' + pid ] = {'true_color': img_thumb}
    torch.save(image_thumbnails_train_dict, out_root_dict)
    print('Saving the ZY3_Trainset thumbnails to the dict file: ', out_root_dict)

if __name__=='__main__':

    import pytz  # change timezone
    current_date = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing
    datehour_mark = '%04d%02d%02d_%02d' % (current_date.year, current_date.month, current_date.day, current_date.hour)
    print('Onet_pid: ',     os.getpid())
    print('current w_dir ', os.getcwd())

    # config = conf_model.generate_config(
    #     './configs/test_pre_processing_on_zy3_testset_20240321.yml', dataset_name='zy3')

    config = conf_model.generate_config(
        './configs/train_onet_20250407.yml', dataset_name='zy3')
    print('checkpoint_directory:\n ', config.out_root)

    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    #config.start_time = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing, for computing total training time.

    print(uti.config_to_str(config))  # print configuration to log file.
    log_file_name = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file_name, encoding='utf-8', level=logging.INFO)


    zy3_train_image_dict      = cloud_model_zy3.prepare_cloud_traindata(config)
    zy3_test_image_label_dict = cloud_model_zy3.prepare_cloud_testdata(config)
    zy3_aug_train_dataloader  = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_image_dict,      bsu=False, baug=True) # augmentation on images with no labels
    zy3_test_dataloader       = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_test_image_label_dict, bsu=True,  baug=False)

    # with output channel=2 for outc layer, should set n_classes=2 in onet_vanilla_outc.py
    onet = onet_vanilla_model.Onet(in_chns=3, binit=True, bshare=True)  # in revised tip, onet_vanilla use RGB.
    logging.info(uti.config_to_str(config))
    logging.info(uti.count_parameters(onet, bverbose=False))  # count paramermters without detail names.
    onet.to(config.device)

    # model_file = '/root/tip_onet_revision_gz16g/checkpoint/onet_vanilla_unsupervised_snow_feature_alignment/'\
    #               'onet_vanilla_unsupervised_zy3trainset178_testset50_acc_0.8799_miou_0.7130_20240308_11.pytorch'
    #model_file ='/root/tip_onet2.0_202403/checkpoint/zy3/onet_vanilla/onet_vanilla_zy3_epoch300_20240606_16.pytorch'
    model_file = os.path.join(config.out_root, config.model_file)
    dict_info = torch.load(model_file, map_location=lambda storage, loc: storage)
    onet.load_state_dict(dict_info["net"])
    config.last_epoch = dict_info["save_epoch"]
    config.save_epoch = config.last_epoch
    #uti_zy3_test.assign_fg_mark_v2(config, onet, zy3_test_dataloader)  # assign the fg_mark to onet.

    #prepare_train_thumbnails()
    zy3_divided_test_ids_dict = uti_zy3_test.get_divided_test_list()
    choose_test_preprocess(config, onet) #saving test_image_label_dict with the best pre-processing option.
    #classified_preprocess(config, onet, zy3_divided_test_ids_dict) # determine the pre-processing option for each test image.
    #exit(0)

    pre_process_zy3_test_image_label_dict = torch.load('/root/datasets/thumbnail224/zy3_thumbnail224_test_label_dict50_bestACC_preprocess.pt')
    #pre_process_zy3_test_image_label_dict = torch.load('/root/datasets/thumbnail224/zy3_thumbnail224_test_label_dict50_v3_[haze_remove_enhance].pt')
    zy3_pre_proc_test_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, pre_process_zy3_test_image_label_dict, bsu=True,baug=False)
    # test_on_zy3_testset_thin_cloud(config, onet, aug_zy3_test_dataloader, bias=0)
    zy3_divided_test_ids_dict = uti_zy3_test.get_divided_test_list()  # get the divided test list for monitoring results of different test sets.
    #uti_zy3_test.test_on_zy3_nail_v3(config, onet, zy3_pre_proc_test_dataloader, divided_list_dict=zy3_divided_test_ids_dict,bsave_excel=True)
    uti_zy3_test.save_zy3_test_results_to_excel(config, onet, zy3_pre_proc_test_dataloader, divided_list_dict=zy3_divided_test_ids_dict,bsave_excel=False)