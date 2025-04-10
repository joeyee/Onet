'''
Train Onet_vanilla on ZY3 cloud segmentation datasets.
Based on the file 'Onet_vanilla_zy3_snow_alignment_20240305.py'

Created by ZhouYi@Linghai_Dalian on 20240606
'''


import os
import pytz  # change timezone
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime
import torch
import torch.nn as nn
import utils_20231218 as uti
import configs.config_tip2022_20230411             as conf_model
import dataloader.zy3_cloud_thumbnailv5_20240304   as cloud_model_zy3
import uti_zy3_test_20240123 as uti_zy3_test
import Onet_vanilla_20240606                       as onet_vanilla_model
# import onet_ablation_RSN_20230626                  as onet_vanilla_rsn_model
# import Onet_ConvNext_20240607                      as onet_convnext_model
# import Onet_vanilla_outc_20240609                  as onet_vanilla_outc_model
# import Onet_ConvNeXt_outc_20240609                 as onet_convnext_outc_model
# import Onet_Trans_Unet_20240613                    as onet_transunet_model
# import Onet_Swin_Transformer_20240615              as onet_swin_model
#import Onet_vanilla_outc_20240305                  as onet_vanilla_outc
import select_trainset_for_correct_clouds_20240307 as select_trainset
import torch.nn.functional as F
import copy
import logging
import glob
import math

from tqdm import tqdm
torch.manual_seed(1981)
np.random.seed(1981)
torch.set_default_dtype(torch.float32)

# bright and thin cloud images in zy3 trainset
bright_terrain_only_tid_list = ['1712110579', '1712119384', '17012277836044602',
                                '14120165633954060', '16102400785553324',
                                '17040744346354862', '17042537416415819',
                                '14112183753944996', '14112058503919781',
                                '17032328336275058', '17041101236382351',
                                '17042778266418650', '17032348016275067',
                                '17020878486096639', '1706157731',
                                '17101849847322975', '14112058503919781',
                                '17091422397172296',  # not find 1709142239712296 in trainset
                                '15012121584084790', '14121797223988205', '17120177657504539',
                                '17042552596415577', '1710177084',
                                '14120165633954060', '1210290160586232',  # ice/snow land
                                '1711213249', '1712019369', '1712011771', '1712115356',
                                '1712115356', '1712118502', '1609232855', '1609230301', '1712118687', '1712010073',
                                '1711218658', '1712125418', '1710174747', '1609238908',  # sand/rock land
                                '1712125418', '1712112118'  # suburb
                                ]
thin_clouds_only_tid_list = ['1711210256', '1711211564', '1711212921', '1711210256',
                             '1702105821', '1710171813', '17062222776751076', '1712043142',
                             '1710172901', '1706156981', '1706159113', '17062206586751086',
                             '1706150953', '1702105574', '1712075643', '1712077273',
                             '1711215376', '17042089736367046', '17041105826425179', '1706158902',
                             ]


def freaze_model(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreaze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def train_onet_unsupervised_on_zy3(config, onet, train_dataloader, test_dataloader):
    '''train onet zy3 unsupervisedly.'''
    print('Start training %s in %d epoches' % (config.model_name, config.epoch_nums))
    if config.restart:
        dict_info = torch.load(config.model_file, map_location=lambda storage, loc: storage)
        onet.load_state_dict(dict_info["net"])
        config.last_epoch = dict_info["save_epoch"]
        config.save_epoch = config.last_epoch
        test_loss, best_acc, best_miou = uti_zy3_test.test_on_zy3_nail(config, onet, test_dataloader, draw_batch=True, draw_all=False)
    else:
        config.save_epoch = 0
        best_acc, best_miou = (0., 0.)

    # opt = torch.optim.Adam(onet.parameters(), lr=1e-5 / 2, betas=(0.9, 0.999),
    #                          eps=1e-08, weight_decay=0, amsgrad=False)  # maximize=False)
    opt = torch.optim.Adam(onet.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # maximize=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=300, T_mult=2, eta_min=0.000001, last_epoch=-1)
    # best_acc, best_miou = (0., 0.)
    loss_list = []  # training record for all epoches
    test_loss_list = []  # test loss for all epoches.
    acc_list = []  # testing record
    miou_list = []  # testing record
    tr_fig, tr_axs   = plt.subplots(figsize=(6, 5))  # show train loss
    res_fig, res_axs = plt.subplots(figsize=(6, 5))  # show test loss, acc, miou

    for epoch in range(config.epoch_nums):
        onet.train()
        loss_batch_list = []
        for X, img_ids in train_dataloader:  # train_labels is not use.
            onet.zero_grad()

            X = X.to(config.device)
            # Lt, St = onet.topu(X) # in this version softmax for topu is inside topu for two_class output.
            # Vt = St[:, 1, :, :].unsqueeze(1)
            # Ld, Sd = onet.topu(1-X)  # negative_augmentation for unsupervised learning.
            # Vd = Sd[:, 1, :, :].unsqueeze(1)
            # S = onet.softmax(torch.cat([Vt, Vd], dim=1)) #final score map for two classes.
            # Sf = S[:, 0, :, :].unsqueeze(dim=1)
            # Sb = S[:, 1, :, :].unsqueeze(dim=1)
            # loss = onet.compute_loss(Lt, Sf, Ld, Sb)
            Lt, Vt, Ld, Vd, S = onet(X)  # send rgb to onet for convnext backbone
            St = S[:, 0, :, :].unsqueeze(dim=1)
            Sd = S[:, 1, :, :].unsqueeze(dim=1)
            loss = onet.compute_loss(Lt, St, Ld, Sd)

            loss.backward()
            opt.step()
            loss_batch_list.append(loss.item())

        loss_epoch = np.mean(np.array(loss_batch_list))
        loss_list.append(loss_epoch)

        # if epoch % 100 == 0 and epoch > 0:  # decay the learning rate every 100 epoches
        #     opt.param_groups[0]['lr'] *= 0.5
        scheduler.step()
        config.last_epoch = epoch

        test_loss_epoch, acc_epoch, miou_epoch, dr_epoch, far_epoch =uti_zy3_test.test_on_zy3_nail(config, onet, test_dataloader, draw_batch=False, draw_all=False)
        acc_list.append(acc_epoch)
        miou_list.append(miou_epoch)

        print("%s===Epoch: %04d, Training loss: %.2E, lr: %.2E,miou %.4f acc %.4f %s===" %
              (config.model_name, epoch, loss_epoch, opt.param_groups[0]['lr'], miou_epoch, acc_epoch, datetime.now(pytz.timezone('Asia/Shanghai'))))
        logging.info("%s===Epoch: %04d, Training loss: %.2E, lr: %.2E,miou %.4f acc %.4f %s===" %
                (config.model_name, epoch, loss_epoch, opt.param_groups[0]['lr'], miou_epoch, acc_epoch, datetime.now(pytz.timezone('Asia/Shanghai'))))
        #if best_loss > loss_epoch:
        #if best_acc < acc_epoch and acc_epoch>0.85:      # saving the model with best performance
        if epoch == config.epoch_nums - 1 or epoch == 300:# saving the model when reaching the final epoch.
            best_acc = acc_epoch
            print('Saving net dict at epoch %d' % epoch)
            logging.info('Saving net dict at epoch %d' % epoch)
            save_dict_info = {"net": copy.deepcopy(onet.state_dict()), 'save_epoch':epoch}  # saving the model with best performance
            logging.info('Saving net dict at epoch %d' % epoch)
            #torch.save(save_dict_info, os.path.join(config.out_root, "%s_%s.pytorch" % (config.model_name, datehour_mark)))
            ##---saving more files for model averaging---##
            torch.save(save_dict_info, os.path.join(config.out_root, "%s_epoch%d_%s.pytorch" % (config.model_name, epoch, config.datehour_mark)))
            model_file_path = os.path.join(config.out_root, "%s_%s.pytorch" % (config.model_name, config.datehour_mark))
            print('saving model %s' % model_file_path)
            logging.info('saving model %s' % model_file_path)
            config.save_epoch = epoch
            uti_zy3_test.test_on_zy3_nail(config, onet, test_dataloader, draw_batch=True, draw_all=False) # draw batch samples for current best model.
        # draw train loss
        tr_axs.clear()
        tr_axs.plot(loss_list,      'r'  , label='train_loss')
        tr_axs.plot(test_loss_list, 'g-.', label='test_loss')
        # tr_axs.plot(jsd_list,     'g-.', label='jsd')
        tr_axs.legend()
        tr_fig.savefig(os.path.join(config.out_root, "%s_train_loss_%s.png" % (config.model_name, config.datehour_mark)))

        res_axs.clear()
        res_axs.plot(acc_list, 'r', label='acc')
        res_axs.plot(miou_list, 'g-.', label='miou')
        res_axs.legend()
        res_fig.savefig(os.path.join(config.out_root, "%s_test_acc_miou_%s.png" % (config.model_name, config.datehour_mark)))

    onet.load_state_dict(save_dict_info["net"])
    uti_zy3_test.test_on_zy3_nail(config, onet, test_dataloader, draw_batch=False,draw_all=True)  # test the best model and draw all the results.
    # another way to check which pid using GPU: (sudo fuser -v /dev/nvidia* and kill pid)
    #del onet
    torch.cuda.empty_cache()
    gc.collect()
    print('Finish training %s' % config.model_name)
    logging.info('Finish training %s' % config.model_name)
    return onet

def test_model_performance(config):
    path = os.path.join(config.dataset_root, 'zy3_thumbnail224_test_label_dict50_bestACC_preprocess.pt')
    test_img_dict = torch.load(path)
    print(''.join(['The number of samples in the test dataset is ', str(len(test_img_dict))]))
    zy3_test_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, test_img_dict, bsu=True, baug=False)

    # config.out_root= '/root/tip_onet2.0_202403/checkpoint/zy3/onet_vanilla/'
    #onet = onet_vanilla_model.Onet(in_chns=3, binit=True, bshare=True)  # in revised tip, onet_vanilla use RGB.

    # config.out_root = '/root/tip_onet2.0_202403/checkpoint/zy3/onet_vanilla_outc'
    # onet = onet_vanilla_outc_model.Onet(in_chns=config.input_chn, n_classes=config.gt_k)

    #config.out_root= '/root/tip_onet2.0_202403/checkpoint/onet_convnext_zy3_2.0_unsupervised/'
    #onet = onet_convnext_model.Onet_ConvNeXt(in_chns=3, binit=True, bshare=True)

    # config.out_root = '/root/tip_onet2.0_202403/checkpoint/zy3/onet_convnext'
    # config.out_root= '/root/tip_onet2.0_202403/checkpoint/zy3/onet_convnext_outc'
    # onet = onet_convnext_outc_model.Onet_ConvNeXt(in_chns=config.input_chn, n_classes=config.gt_k, bshare=True)

    # import Onet_Trans_Unet_config_20230409 as onet_trans_conf #special configuration for swin_unet
    # trans_config = onet_trans_conf.get_config()
    # config.out_root = "/root/tip_onet2.0_202403/checkpoint/zy3/onet_transunet"
    # onet = onet_transunet_model.Onet_Trans_Unet(trans_config, in_chns=config.input_chn, n_classes=2, binit=False, bshare=True)


    import Onet_Swin_config_20230408 as onet_swin_conf #special configuration for swin_unet
    swin_config = onet_swin_conf.get_config()
    config.out_root= '/root/tip_onet2.0_202403/checkpoint/zy3/onet_swin'
    onet = onet_swin_model.Onet_Swin_Unet(swin_config, in_chns=3, n_classes=2, bshare=True)



    model_files = glob.glob(os.path.join(config.out_root, '*.pytorch'))
    model_files.sort()
    for model_file in model_files:
        print('=== test model %s ===' % model_file)
        print('=== on dataset %s ===' % path)
        onet.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['net'])
        # model_file = '/root/tip_onet2.0_202403/checkpoint/zy3/onet_convnext/onet_convnext_zy3_epoch300_20240607_15.pytorch'
        # onet.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)["net"])
        onet.to(config.device)
        test_loss, acc, miou, pd, far = uti_zy3_test.test_on_zy3_nail(config, onet, zy3_test_dataloader,draw_batch=False, draw_all=False)

        #test_loss, best_acc, best_miou = uti_zy3_test.test_on_zy3_nail_v2(config, onet, zy3_test_dataloader, draw_batch=False,draw_all=False)
        print('acc %.4f, miou %.4f, pd %.4f, far %.4f' % (acc, miou, pd, far))

def train_rsn_model(): #train the random_sampling_mode for ablation study
    config = conf_model.generate_config('./configs/train_onet_vallina_20240606.yml', dataset_name='zy3')
    config.out_root = '/root/tip_onet2.0_202403/checkpoint/zy3/onet_vanilla_rsn'

    # config = conf_model.generate_config('./configs/train_onet_vallina_20240606.yml', dataset_name='Rayleigh')
    # config.out_root = '/root/tip_onet2.0_202403/checkpoint/sim_clutter/onet_vanilla_rsn'

    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    current_date = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing
    datehour_mark = '%04d%02d%02d_%02d' % (current_date.year, current_date.month, current_date.day, current_date.hour)
    config.datehour_mark = datehour_mark
    print('Onet_pid: ', os.getpid())
    print('current w_dir ', os.getcwd())
    print('checkpoint_directory:\n ', config.out_root)
    config.start_time = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing, for computing total training time.
    print(uti.config_to_str(config))  # print configuration to log file.

    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    print(uti.config_to_str(config))  # print configuration to log file.
    logging.info(uti.config_to_str(config))  # print configuration to log file.
    # config.log_file = config.log_file.split('.')[0] + '%s.log' % (datehour_mark)  # add time stamp to log file name
    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

    ##do not use the whole training set of zy3
    # zy3_train_image_dict = cloud_model_zy3.prepare_cloud_traindata(config)
    # zy3_aug_train_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_image_dict, bsu=False, baug=True)  # augmentation on images with no labels
    zy3_test_image_label_dict = cloud_model_zy3.prepare_cloud_testdata(config)

    zy3_train_clouds_no_snow_dict, zy3_train_bright_terrain_only_dict = select_trainset.divide_zy3_traindata(config)
    zy3_aug_train_no_snow_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_clouds_no_snow_dict,
                                                                                 bsu=False, baug=True)

    zy3_test_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_test_image_label_dict, bsu=True,
                                                                    baug=False)  # no augmentation on test set, but get labels.

    onet = onet_vanilla_rsn_model.Onet(in_chns=config.input_chn, bshare=True)  # using weight-sharing modes

    print(uti.count_parameters(onet, bverbose=False))
    logging.info(uti.count_parameters(onet, bverbose=False))
    onet.to(config.device)
    zy3_divided_test_ids_dict = uti_zy3_test.get_divided_test_list()  # get the divided test list for monitoring results of different test sets.
    train_onet_unsupervised_on_zy3(config, onet, zy3_aug_train_no_snow_dataloader, zy3_test_dataloader)
def train_lhd_model(): # train the onet1.0 version on zy3
    '''twins cnn_unet mode'''
    config = conf_model.generate_config('./configs/train_onet_vallina_20240606.yml', dataset_name='zy3')
    config.out_root = '/root/tip_onet2.0_202403/checkpoint/zy3/onet_vanilla_twins'
    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    current_date = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing
    datehour_mark = '%04d%02d%02d_%02d' % (current_date.year, current_date.month, current_date.day, current_date.hour)
    config.datehour_mark = datehour_mark
    print('Onet_pid: ', os.getpid())
    print('current w_dir ', os.getcwd())
    print('checkpoint_directory:\n ', config.out_root)
    config.start_time = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing, for computing total training time.
    print(uti.config_to_str(config))  # print configuration to log file.

    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    print(uti.config_to_str(config))  # print configuration to log file.
    logging.info(uti.config_to_str(config))  # print configuration to log file.
    # config.log_file = config.log_file.split('.')[0] + '%s.log' % (datehour_mark)  # add time stamp to log file name
    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

    ##do not use the whole training set of zy3
    # zy3_train_image_dict = cloud_model_zy3.prepare_cloud_traindata(config)
    # zy3_aug_train_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_image_dict, bsu=False, baug=True)  # augmentation on images with no labels
    zy3_test_image_label_dict = cloud_model_zy3.prepare_cloud_testdata(config)

    zy3_train_clouds_no_snow_dict, zy3_train_bright_terrain_only_dict = select_trainset.divide_zy3_traindata(config)
    zy3_aug_train_no_snow_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_clouds_no_snow_dict, bsu=False, baug=True)


    zy3_test_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_test_image_label_dict, bsu=True, baug=False)  # no augmentation on test set, but get labels.


    onet = onet_vanilla_model.Onet(in_chns=config.input_chn, bshare=False) #using twin modes

    print(uti.count_parameters(onet, bverbose=False))
    logging.info(uti.count_parameters(onet, bverbose=False))
    onet.to(config.device)
    zy3_divided_test_ids_dict = uti_zy3_test.get_divided_test_list()  # get the divided test list for monitoring results of different test sets.
    train_onet_unsupervised_on_zy3(config, onet, zy3_aug_train_no_snow_dataloader, zy3_test_dataloader)

def train_outc_model():
    '''
    Verify the output head with outc layer. conv2d(64,2) to get the score map.
    :param config:
    :return:
    '''
    import Onet_ConvNeXt_Block_20240331 as onet_convnext_block #outc layer is added in the block.
    import Onet_ConvNeXt_zy3_v2_20240331 as onet_convnext_outc_app
    current_date = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing
    # config = conf_model.generate_config('configs/train_onet_convnext_20240607.yml', dataset_name='zy3')
    # config.out_root = '/root/tip_onet2.0_202403/checkpoint/zy3/onet_convnext_outc'
    # config.out_root = '/root/tip_onet2.0_202403/checkpoint/zy3/onet_vanilla_outc'
    #config = conf_model.generate_config('./configs/train_onet_transunet_20240613.yml', dataset_name='zy3')
    config = conf_model.generate_config('./configs/train_onet_swin_20240615.yml', dataset_name='zy3')
    config.model_name += '_outc'
    current_date = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing
    datehour_mark = '%04d%02d%02d_%02d' % (current_date.year, current_date.month, current_date.day, current_date.hour)
    config.datehour_mark = datehour_mark
    print('Onet_pid: ', os.getpid())
    print('current w_dir ', os.getcwd())
    print('checkpoint_directory:\n ', config.out_root)
    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    config.start_time = datetime.now(
        pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing, for computing total training time.
    print(uti.config_to_str(config))  # print configuration to log file.

    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    print(uti.config_to_str(config))  # print configuration to log file.
    logging.info(uti.config_to_str(config))  # print configuration to log file.
    # config.log_file = config.log_file.split('.')[0] + '%s.log' % (datehour_mark)  # add time stamp to log file name
    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

    # zy3_train_dataloader, zy3_test_dataloader = cloud_model_zy3.cloud_dataloader(config)
    zy3_train_image_dict = cloud_model_zy3.prepare_cloud_traindata(config)
    zy3_test_image_label_dict = cloud_model_zy3.prepare_cloud_testdata(config)

    zy3_train_clouds_no_snow_dict, zy3_train_bright_terrain_only_dict = select_trainset.divide_zy3_traindata(config)
    zy3_aug_train_no_snow_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_clouds_no_snow_dict, bsu=False, baug=True)

    zy3_aug_train_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_image_dict, bsu=False,
                                                                         baug=True)  # augmentation on images with no labels
    zy3_test_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_test_image_label_dict, bsu=True,
                                                                    baug=False)  # no augmentation on test set, but get labels.

    ### train and test convnext_outc model #
    # onet = onet_convnext_model.Onet_ConvNeXt(in_chns=config.input_chn, n_classes=config.gt_k, cnx_type=config.type)
    # onet = onet_convnext_outc_model.Onet_ConvNeXt(in_chns=config.input_chn, n_classes=config.gt_k)
    #onet = onet_vanilla_outc_model.Onet(in_chns=config.input_chn, n_classes=config.gt_k)

    ### train and test onet_transunet model #
    # import Onet_Trans_Unet_config_20230409 as onet_trans_conf #special configuration for swin_unet
    # trans_config = onet_trans_conf.get_config()
    # onet = onet_transunet_model.Onet_Trans_Unet(trans_config, in_chns=config.input_chn, n_classes=2, binit=False, bshare=True)

    ### train and test onet_swin model #
    import Onet_Swin_config_20230408 as onet_swin_conf #special configuration for swin_unet
    swin_config = onet_swin_conf.get_config()
    onet = onet_swin_model.Onet_Swin_Unet(swin_config, in_chns=3, n_classes=2, bshare=True)
    onet.to(config.device)

    print(uti.count_parameters(onet, bverbose=False))
    logging.info(uti.count_parameters(onet, bverbose=False))
    onet.to(config.device)

    zy3_divided_test_ids_dict = uti_zy3_test.get_divided_test_list()  # get the divided test list for monitoring results of different test sets.
    #onet_convnext_outc_app.train_onet_unsupervised_on_zy3(config, onet, zy3_aug_train_dataloader, zy3_test_dataloader)
    train_onet_unsupervised_on_zy3(config, onet, zy3_aug_train_no_snow_dataloader, zy3_test_dataloader)


#nohup python -u Onet_convnext_zy3_20231219.py  > ./checkpoint/cloud_zy3_onet_convnext_tiny_imagenet22k/cloud_zy3_onet_convnext_tiny_22k_20231221.log &
#nohup python -u Onet_convnext_zy3_20231219.py  > ./checkpoint/cloud_zy3_onet_convnext_base_imagenet22k/cloud_zy3_onet_convnext_base_22k_20231221.log &
#nohup python -u Onet_convnext_zy3_20231219.py  > ./checkpoint/cloud_zy3_onet_Ex_convnext_tiny_none/cloud_zy3_onet_Ex_convnext_tiny_20231221.log &
#nohup python -u Onet_convnext_zy3_20231219.py  > ./checkpoint/cloud_zy3_onet_Ex_convnext_base_none/cloud_zy3_onet_Ex_convnext_base_20231222.log &
'''
nohup python -u Onet_vanilla_zy3_20231226.py  > ./checkpoint/zy3_onet_vanilla_outc1class_unsupervised/onet_vanilla_outc1_unsupervised.log &
nohup python -u Onet_vanilla_zy3_20231226.py > ./checkpoint/zy3_onet_vanilla_outc2class_unsupervised/zy3_onet_vanilla_outc2class_20231230.log &
nohup python -u Onet_vanilla_zy3_20240126.py > nohup_onet_unsupervised_trained_on_zy3.out &
nohup python -u Onet_vanilla_zy3_snow_alignment_20240305.py > nohup.out &
nohup python -u Onet_vanilla_zy3_snow_alignment_20240305.py > nohup_un_supervised_[haze_enhance_remove]_v2.out &
nohup python -u Train_Onet_on_zy3_20240606.py > Train_Onet_vanilla_on_zy3_20240606.out &
nohup python -u Train_Onet_on_zy3_20240606.py > Train_Onet_convnext_on_zy3_20240607.out &
nohup python -u Train_Onet_on_zy3_20240606.py > Train_Onet_convnext_outc_on_zy3_20240609.out &
nohup python -u Train_Onet_on_zy3_20240606.py > Train_Onet_vanilla_outc_on_zy3_20240609.out &
nohup python -u Train_Onet_on_zy3_20240606.py > Train_Onet_transunet_on_zy3_20240614.out &
nohup python -u Train_Onet_on_zy3_20240606.py > Train_Onet_swin_on_zy3_20240615.out &
nohup python -u Train_Onet_on_zy3_20240606.py > Train_Onet_vanilla_twins_on_zy3_20240617.out &
nohup python -u Train_Onet_on_zy3_20240606.py > ./checkpoint/zy3/onet_vanilla_rsn/Train_Onet_vanilla_rsn_on_zy3_20240618.out &
'''
if __name__=='__main__':

    print('Onet_pid: ',     os.getpid())
    print('current w_dir ', os.getcwd())
    # train_outc_model()
    # exit(0)
    # train_lhd_model()
    # exit(0)
    # train_rsn_model()
    # exit(0)

    current_date = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing
    datehour_mark = '%04d%02d%02d_%02d' % (current_date.year, current_date.month, current_date.day, current_date.hour)
    
    config = conf_model.generate_config('./configs/train_onet_20250407.yml', dataset_name='zy3') # train onet for published tip on 2025-04-09
    #config = conf_model.generate_config('./configs/train_onet_vallina_20240606.yml', dataset_name='zy3') # train onet for ablation when revised tip in 2024
    #config = conf_model.generate_config('./configs/train_onet_convnext_20240607.yml', dataset_name='zy3')
    config.datehour_mark = datehour_mark
    print('checkpoint_directory:\n ', config.out_root)
    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    config.start_time = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing, for computing total training time.

    print(uti.config_to_str(config))  # print configuration to log file.
    # config.log_file = config.log_file.split('.')[0] + '%s.log' % (datehour_mark)  # add time stamp to log file name
    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

    # test_model_performance(config)
    # exit(0)

    # zy3_train_dataloader, zy3_test_dataloader = cloud_model_zy3.cloud_dataloader(config)
    zy3_train_image_dict      = cloud_model_zy3.prepare_cloud_traindata(config)
    zy3_test_image_label_dict = cloud_model_zy3.prepare_cloud_testdata(config)

    zy3_aug_train_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_image_dict, bsu=False, baug=True) # augmentation on images with no labels
    zy3_test_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_test_image_label_dict, bsu=True,  baug=False) # no augmentation on test set, but get labels.


    # with output channel=2 for outc layer, should set n_classes=2 in onet_vanilla_outc.py
    onet = onet_vanilla_model.Onet(in_chns=3, binit=True, bshare=True)  # in revised tip, onet_vanilla use RGB.
    #onet = onet_convnext_model.Onet_ConvNeXt(in_chns=3, binit=True, bshare=True)  # in revised tip, onet_vanilla use RGB.

    logging.info(uti.config_to_str(config))
    logging.info(uti.count_parameters(onet, bverbose=False))  # count paramermters without detail names.
    onet.to(config.device)

    #train onet unsupervised on the zy3_train_clouds with out snow set from scratch.
    zy3_train_clouds_no_snow_dict, zy3_train_bright_terrain_only_dict = select_trainset.divide_zy3_traindata(config)
    zy3_aug_train_no_snow_dataloader = cloud_model_zy3.cloud_dataloader_via_dict(config, zy3_train_clouds_no_snow_dict, bsu=False, baug=True)
    onet = train_onet_unsupervised_on_zy3(config, onet, zy3_aug_train_no_snow_dataloader, zy3_test_dataloader) #best performance at epoch 200, acc 0.89, miou 0.78

    #get the results on divided test sets.
    zy3_divided_test_ids_dict = uti_zy3_test.get_divided_test_list()  # get the divided test list for monitoring results of different test sets.
    uti_zy3_test.save_zy3_test_results_to_excel(config, onet, zy3_test_dataloader, divided_list_dict=zy3_divided_test_ids_dict,bsave_excel=True)



    # model_file = '/root/tip_onet_revision_gz16g/checkpoint/onet_vanilla_unsupervised_snow_feature_alignment/'\
    #               'onet_vanilla_unsupervised_zy3trainset178_testset50_acc_0.8799_miou_0.7130_20240308_11.pytorch'



