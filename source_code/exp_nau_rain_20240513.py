'''
Test Onet2.0(weight share mode) on the NAU rain dataset.

Created by ZhouYi@Linghai_Dalian on 2024/05/13
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import utils_20231218 as uti
import pickle
from datetime import datetime
from skimage.transform import resize
from tqdm import tqdm

#import datasets.dsb_nucleus4onet_20230306 as dsb_model   # load dsb_datasets.
import configs.config_tip2022_20230411    as conf_model  # load configuration from '.yml' file.
#import evaluation_20230429                as eval_model  # load evaluation function.
import dataloader.simbg4onet_20230209       as simbg_model # load simbg_datasets.
#import Onet_v1_20230428                      as onet_simbg_model  # load Onet1.0(twins) model.
import Onet_vanilla_20240606               as onet_rev_model # weight share mode
import dataloader.nau_rain_20230523         as naurain_model
#import cfar_segmentation_200527           as cfar_model
import pandas as pd
import gc
import glob
import logging

#import InfoSeg_Simbg_20230527  as infoseg_model
import torchvision.transforms  as transforms

torch.manual_seed(1981)
np.random.seed(1981)
torch.set_default_dtype(torch.float32)

def test_naurain_onet(config, onet,  test_loader):
    onet.eval()
    acc_batch_list = []
    miou_batch_list = []
    dr_batch_list = []
    far_batch_list = []
    psnr_batch_list = []
    snr_batch_list  = []
    with torch.no_grad():
        for names, X, label in test_loader:
            X = X.to(config.device)
            label = label.to(config.device)
            Lt, Vt, Ld, Vd, S = onet(X)
            Vt = uti.tensor_normal_per_frame(Vt)
            Vd = uti.tensor_normal_per_frame(Vd)
            raw_label = onet.predict_label(S)  # 0 is foreground, 1 is background  in the case of strong foreground (e.g. fg is cloud).
            pred_label = uti.re_assign_label(raw_label,label)  # re-assign the label to make sure the foreground is 1
            batch_acc, batch_miou, batch_dr, batch_far, batch_tiou = uti.evaluate_nau_segmentation_v2(pred_label, label)
            acc_batch_list.append(batch_acc)
            miou_batch_list.append(batch_miou)
            dr_batch_list.append(batch_dr)
            far_batch_list.append(batch_far)
            pred_t = Vt.squeeze(dim=1)
            pred_d = Vd.squeeze(dim=1)
            #uti.show_segmentation(X, pred_label, label,  'Onet', config)
            uti.show_unet_adversarial_v2(X, pred_t, pred_d, label, pred_label, 'onet2.0_wtshare', config)
            input_psnr, input_snr = uti.get_psnr(X.squeeze(dim=1), label)
            psnr_batch_list.append(input_psnr)
            snr_batch_list.append(input_snr)
        #
    acc = np.array(acc_batch_list).mean()  # overall accuracy
    miou = np.array(miou_batch_list).mean()  # mean iou
    dr = np.array(dr_batch_list).mean()  # detection rate
    far = np.array(far_batch_list).mean()  # false alarm rate
    psnr = np.array(psnr_batch_list).mean()
    snr  = np.array(snr_batch_list).mean()
    return acc, miou, dr, far, psnr, snr

def fig_onet2_stage_vs_onet_weight_share_naurain_revision(config):
    '''
    Draw the figure for the comparison of CFAR and two stage onet.
    :param config:
    :return:
    '''
    onet1st = onet_simbg_model.Onet(binit=True)
    onet1st.to(config.device)
    onet1st.load_state_dict(torch.load(config.model_file_onet1st, map_location=lambda storage, loc: storage)['net'])

    onet2nd = onet_simbg_model.Onet(binit=True)
    onet2nd.to(config.device)
    onet2nd.load_state_dict(torch.load(config.model_file_onet2nd, map_location=lambda storage, loc: storage)['net'])

    onet1st.eval()  # set the model to evaluation mode, without this line, the output will be different, more false alarms.
    onet2nd.eval()

    onet_wtshare = onet_rev_model.Onet(binit=True, bshare=True)
    onet_wtshare.to(config.device)
    onet_wtshare.load_state_dict(torch.load(config.model_revison_file, map_location=lambda storage, loc: storage)['net'])
    onet_wtshare.eval() # set the model to evaluation mode, without this line, the output will be different

    psnrs = [1,3,5,7,9]
    fig, axs = plt.subplots(5, 4, figsize=(12, 12*5/4), dpi=300, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plt.rcParams["font.serif"] = ["Times New Roman"]
    axs[0, 0].set_title('Input frame',fontsize=12)
    axs[0, 1].set_title('Ground truth',fontsize=12)
    #axs[0, 2].set_title('CFAR',fontsize=12)
    #axs[0, 3].set_title('Two-stage Onet',fontsize=12)
    onetws_far_list = []
    onet2_far_list = []
    test_dataloader = naurain_model.make_nau_rain_dataloader(config)

    with torch.no_grad():
        for names, X1, label in test_dataloader: # batchsize=20, one batch contains 20 samples.
            kval = 1.3
            kval = 2.0  # far = 0.03
            cfar = cfar_model.CFAR(kval=kval, nref=16, mguide=8)

            label = label.to(config.device)
            X1 = X1.to(config.device)
            Lt_ws, Vt_ws, Ld_ws, Vd_ws, S_ws = onet_wtshare(X1)  # onet predict.
            raw_label_ws = onet_wtshare.predict_label(S_ws)  # 1 means Vd>Vt (fg is Vd), 0 means Vt>Vd (fg is Vt).
            pred_label_ws = uti.re_assign_label(raw_label_ws, label)  # re-assign the label according to the gt label
            acc_ws, miou_ws, dr_ws, far_ws, tiou_ws = uti.evaluate_nau_segmentation_v2(pred_label_ws, label)
            onetws_far_list.append(far_ws)
            label = label.to(config.device)
            X1 = X1.to(config.device)
            Lt1, Vt1, Ld1, Vd1, S1 = onet1st(X1)  # onet predict.
            raw_label1 = onet1st.predict_label(S1)  # 1 means Vd>Vt (fg is Vd), 0 means Vt>Vd (fg is Vt).
            pred_label1 = uti.re_assign_label(raw_label1, label)  # re-assign the label according to the gt label
            if torch.equal(raw_label1, pred_label1):  # if the label is not changed, Vd represents the foreground
                fg1 = Vd1
                bg1 = Vt1
            else:  # if the label is changed, Vt represents the foreground
                fg1 = Vt1
                bg1 = Vd1

            X2 = uti.tensor_normal_per_frame(fg1)
            Lt2, Vt2, Ld2, Vd2, S2 = onet2nd(X2)
            raw_label2 = onet2nd.predict_label(S2)  # 0 is foreground, 1 is background  in the case of normalized X'
            pred_label2 = uti.re_assign_label(raw_label2, label)  # re-assign the label according to the gt label
            acc2, miou2, dr2, far2, tiou2 = uti.evaluate_nau_segmentation_v2(pred_label2, label)
            onet2_far_list.append(far2)

            t_letters = ['d', 'f', 'i', 'j', 'k']
            #rmfont = {'fontname':'Times New Roman'}
            # // write your code related to basemap here
            # plt.title('title',**csfont)
            for i,name in enumerate(names): # index in one batch
                letter = name.split('_')[2]
                if letter in t_letters:
                    aid = t_letters.index(letter) # index in the figure
                    subimage = X1[i, 0, :, :].cpu().numpy()

                    gt_label  = label[i, :, :].cpu().numpy()
                    onet_ws_pred = pred_label_ws[i, :, :].cpu().numpy()
                    onet2nd_pred = pred_label2[i, :, :].cpu().numpy()

                    axs[aid, 0].imshow(subimage)
                    axs[aid, 0].text(20, 30, letter.upper(), color='white', fontsize=16)
                    axs[aid, 1].imshow(gt_label)
                    axs[aid, 2].imshow(onet_ws_pred)
                    axs[aid, 3].imshow(onet2nd_pred)
                    for j in range(4):
                        axs[aid, j].set_xticks([])
                        axs[aid, j].set_yticks([])
            axs[0, 2].set_title(r'Onet_WS $P_{fa}=$%.4f' % (np.mean(onetws_far_list)), fontsize=12)
            axs[0, 3].set_title(r'Onet2   $P_{fa}=$%.4f' % (np.mean(onet2_far_list)), fontsize=12)
            # print('CFAR $P_{fa}=$%.2f' % (np.mean(cfar_far_list)))
            # print('Two-stage Onet $P_{fa}=$%.4f' % (np.mean(onet_far_list)))
            # print in latex format

            print('\hline')
            #print('\multirow{2}{*}{%d}' % psnr)
            print('Onet_WS  & %.4f & %.4f & %.4f & %.4f \\\\' % (acc_ws, miou_ws, dr_ws, far_ws))
            print('Onet2 & %.4f & %.4f & %.4f & %.4f \\\\' % (acc2, miou2, dr2, far2))
            plt.show()
            fig.savefig(os.path.join(config.out_root,'exp_naurain_onet2_vs_onet_ws_dfijk.png'), dpi=300, bbox_inches='tight')
def fig_cfar_vs_onet_weight_share_naurain_revision(config):
    '''
    Draw the figure for the comparison of CFAR and two stage onet.
    :param config:
    :return:
    '''
    onet_wtshare = onet_rev_model.Onet(binit=True, bshare=True)
    onet_wtshare.to(config.device)
    onet_wtshare.load_state_dict(torch.load(config.model_file, map_location=lambda storage, loc: storage)['net'])

    onet_wtshare.eval() # set the model to evaluation mode, without this line, the output will be different


    psnrs = [1,3,5,7,9]
    fig, axs = plt.subplots(5, 4, figsize=(12, 12*5/4), dpi=300, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plt.rcParams["font.serif"] = ["Times New Roman"]
    axs[0, 0].set_title('Input frame',fontsize=12)
    axs[0, 1].set_title('Ground truth',fontsize=12)
    #axs[0, 2].set_title('CFAR',fontsize=12)
    #axs[0, 3].set_title('Two-stage Onet',fontsize=12)
    cfar_far_list = []
    cfar_acc_list = []
    cfar_dr_list = []
    cfar_miou_list = []
    onet_far_list = []
    test_dataloader = naurain_model.make_nau_rain_dataloader(config)

    with torch.no_grad():
        for names, X1, label in test_dataloader: # batchsize=20, one batch contains 20 samples.
            kval = 1.3
            kval = 2.0  # far = 0.03
            cfar = cfar_model.CFAR(kval=kval, nref=16, mguide=8)

            label = label.to(config.device)
            X1 = X1.to(config.device)
            Lt1, Vt1, Ld1, Vd1, S1 = onet_wtshare(X1)  # onet predict.
            raw_label1 = onet_wtshare.predict_label(S1)  # 1 means Vd>Vt (fg is Vd), 0 means Vt>Vd (fg is Vt).
            pred_label1 = uti.re_assign_label(raw_label1, label)  # re-assign the label according to the gt label
            acc1, miou1, dr1, far1, tiou1 = uti.evaluate_nau_segmentation_v2(pred_label1, label)

            t_letters = ['d', 'f', 'i', 'j', 'k']
            #rmfont = {'fontname':'Times New Roman'}
            # // write your code related to basemap here
            # plt.title('title',**csfont)
            for i,name in enumerate(names): # index in one batch
                letter = name.split('_')[2]
                if letter in t_letters:
                    aid = t_letters.index(letter) # index in the figure
                    subimage = X1[i, 0, :, :].cpu().numpy()
                    cfar_label= cfar.cfar_seg(subimage)

                    gt_label  = label[i, :, :].cpu().numpy()
                    onet2nd_pred = pred_label1[i, :, :].cpu().numpy()
                    acc_frame, miou_frame, dr_frame, far_frame, tiou_frame = uti.evaluate_nau_segmentation_v2(torch.tensor(cfar_label), torch.tensor(gt_label))
                    print('kval %.2f far %.4f'%(kval, far_frame))
                    cfar_far_list.append(far_frame)
                    cfar_acc_list.append(acc_frame)
                    cfar_dr_list.append(dr_frame)
                    cfar_miou_list.append(miou_frame)
                    onet_far_list.append(far1)
                    axs[aid, 0].imshow(subimage)
                    axs[aid, 0].text(20, 30, letter.upper(), color='white', fontsize=16)
                    axs[aid, 1].imshow(gt_label)
                    axs[aid, 2].imshow(cfar_label)
                    axs[aid, 3].imshow(onet2nd_pred)
                    for j in range(4):
                        axs[aid, j].set_xticks([])
                        axs[aid, j].set_yticks([])
            axs[0, 2].set_title(r'CFAR $P_{fa}=$%.2f' % (np.mean(cfar_far_list)), fontsize=12)
            axs[0, 3].set_title(r'Onet(weight-share) $P_{fa}=$%.4f' % (np.mean(onet_far_list)), fontsize=12)
            # print('CFAR $P_{fa}=$%.2f' % (np.mean(cfar_far_list)))
            # print('Two-stage Onet $P_{fa}=$%.4f' % (np.mean(onet_far_list)))
            # print in latex format
            acc_cfar = np.mean(cfar_acc_list)
            miou_cfar = np.mean(cfar_miou_list)
            dr_cfar = np.mean(cfar_dr_list)
            far_cfar = np.mean(cfar_far_list)
            print('\hline')
            #print('\multirow{2}{*}{%d}' % psnr)
            print('CFAR  & %.4f & %.4f & %.4f & %.4f \\\\' % (acc_cfar, miou_cfar, dr_cfar, far_cfar))
            print('Onet2 & %.4f & %.4f & %.4f & %.4f \\\\' % (acc1, miou1, dr1, far1))
            plt.show()
            fig.savefig(os.path.join(config.out_root,'exp_naurain_cfar_vs_two-stage_Onet_dfijk.png'), dpi=300, bbox_inches='tight')
            #fig.savefig(os.path.join(config.out_root, 'exp_naurain_cfar_vs_two-stage_Onet_islands.png'), dpi=300,bbox_inches='tight')

def fig_onet_weightshare_naurain_islands_revision(config):
    '''
    Draw the figure for the comparison of CFAR and two stage onet.
    :param config:
    :return:
    '''
    onet_wtshare = onet_rev_model.Onet(binit=True, bshare=True)
    onet_wtshare.to(config.device)
    onet_wtshare.load_state_dict(torch.load(config.model_file, map_location=lambda storage, loc: storage)['net'])
    onet_wtshare.eval()

    onet1st = onet_simbg_model.Onet(binit=True)
    onet1st.to(config.device)
    onet1st.load_state_dict(torch.load(config.model_file_onet1st, map_location=lambda storage, loc: storage)['net'])
    onet1st.eval()

    fig, axs = plt.subplots(2, 4, figsize=(12, 12*2/4), dpi=300, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plt.rcParams["font.serif"] = ["Times New Roman"]
    # axs[0, 0].set_title('Input frame',fontsize=12)
    # axs[0, 1].set_title('Ground truth',fontsize=12)
    test_dataloader = naurain_model.make_nau_rain_dataloader(config)
    t_letters = ['1', '2', '3', '4']
    with torch.no_grad():
        for names, X1, label in test_dataloader: # batchsize=20, one batch contains 20 samples.
            label = label.to(config.device) # null label for island roi
            X1 = X1.to(config.device)
            #Lt1, Vt1, Ld1, Vd1, S1 = onet_wtshare(X1)  # onet predict.
            Lt1, Vt1, Ld1, Vd1, S1 = onet1st(X1)  # onet predict.
            raw_label = onet_wtshare.predict_label(S1)  # 1 means Vd>Vt (fg is Vd), 0 means Vt>Vd (fg is Vt).

            for i,name in enumerate(names): # index in one batch
                id      = name[-1]
                aid         = t_letters.index(id) # index in the figure
                subimage    = X1[i, 0, :, :].cpu().numpy()
                pred_fg     = Vd1[i,0, :, :].cpu().numpy()
                pred_label  = raw_label[i, :, :].cpu().numpy()
                axs[0, aid].imshow(subimage)
                axs[0, aid].text(20, 30, 'R'+id, color='white', fontsize=16)
                #axs[1, aid].imshow(pred_label)
                axs[1, aid].imshow(pred_fg)
                axs[1, aid].text(20, 30, 'Fg'+id, color='white', fontsize=16)
                for j in range(2):
                    axs[j,i].set_xticks([])
                    axs[j,i].set_yticks([])
            # axs[0, 2].set_title(r'CFAR $P_{fa}=$%.2f' % (np.mean(cfar_far_list)), fontsize=12)
            # axs[0, 3].set_title(r'Onet(weight-share) $P_{fa}=$%.4f' % (np.mean(onet_far_list)), fontsize=12)
            print('\hline')
            plt.show()
            fig.savefig(os.path.join(config.out_root, 'exp_naurain_onet_weightshare_islands_fg.png'), dpi=300, bbox_inches='tight')

def fig_infoseg_onet_twins_vs_onet_weight_share_naurain_revision(onet_config, infoseg_config):
    '''
    Reviewer questioned that onet is not superior to infoseg. Only the ensemble of onet is better.
    Here in this figure, we plot infoseg, onet_tw, onet_ws and onet2(twin ensemble) for comparison.
    :param config:
    :return:
    '''
    infoseg = infoseg_model.Infoseg(input_channels=infoseg_config.input_chn, K=infoseg_config.gt_k,
                                    height=infoseg_config.input_sz, width=infoseg_config.input_sz)
    infoseg.to(infoseg_config.device)
    infoseg.load_state_dict(torch.load(os.path.join(infoseg_config.out_root, infoseg_config.model_file),
                                       map_location=lambda storage, loc: storage)["net"])
    infoseg.eval()

    onet1st = onet_simbg_model.Onet(binit=True)
    onet1st.to(onet_config.device)
    onet1st.load_state_dict(torch.load(onet_config.model_file_onet1st, map_location=lambda storage, loc: storage)['net'])

    onet2nd = onet_simbg_model.Onet(binit=True)
    onet2nd.to(onet_config.device)
    onet2nd.load_state_dict(torch.load(onet_config.model_file_onet2nd, map_location=lambda storage, loc: storage)['net'])
    onet1st.eval()  # set the model to evaluation mode, without this line, the output will be different, more false alarms.
    onet2nd.eval()

    onet_wtshare = onet_rev_model.Onet(binit=True, bshare=True)
    onet_wtshare.to(onet_config.device)
    onet_wtshare.load_state_dict(torch.load(onet_config.model_file, map_location=lambda storage, loc: storage)['net'])
    onet_wtshare.eval() # set the model to evaluation mode, without this line, the output will be different

    fig, axs = plt.subplots(5, 6, figsize=(14, 14*5/6), dpi=200, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plt.rcParams["font.serif"] = ["Times New Roman"]
    axs[0, 0].set_title('Input frame', fontsize=10)
    axs[0, 1].set_title('Ground truth',fontsize=10)
    # axs[0, 2].set_title('Infoseg',     fontsize=12)
    # axs[0, 3].set_title('Onet_TW',     fontsize=12)
    # axs[0, 4].set_title('Onet_WS',     fontsize=12)
    # axs[0, 5].set_title('Onet2',       fontsize=12)

    infoseg_far_list = []
    onettw_far_list  = []
    onetws_far_list  = []
    onet2_far_list   = []
    test_dataloader = naurain_model.make_nau_rain_dataloader(config)

    with torch.no_grad():
        for names, X1, label in test_dataloader: # batchsize=20, one batch contains 20 samples.

            label = label.to(config.device)
            X1 = X1.to(config.device)

            L, S, V = infoseg(X1)
            info_raw_label = infoseg.get_label(V)
            info_label = uti.re_assign_label(info_raw_label, label)  # re-assign the label according to the gt label
            acc_info, miou_info, dr_info, far_info, tiou_info = uti.evaluate_nau_segmentation_v2(info_label, label)
            infoseg_far_list.append(far_info)

            Lt_ws, Vt_ws, Ld_ws, Vd_ws, S_ws = onet_wtshare(X1)  # onet predict.
            raw_label_ws = onet_wtshare.predict_label(S_ws)  # 1 means Vd>Vt (fg is Vd), 0 means Vt>Vd (fg is Vt).
            pred_label_ws = uti.re_assign_label(raw_label_ws, label)  # re-assign the label according to the gt label
            acc_onet_ws, miou_onet_ws, dr_onet_ws, far_onet_ws, tiou_onet_ws = uti.evaluate_nau_segmentation_v2(pred_label_ws, label)
            onetws_far_list.append(far_onet_ws)

            #onet2 (ensemble two onet_tw modes).
            Lt1, Vt1, Ld1, Vd1, S1 = onet1st(X1)  # onet predict.
            raw_label1 = onet1st.predict_label(S1)  # 1 means Vd>Vt (fg is Vd), 0 means Vt>Vd (fg is Vt).
            pred_label1 = uti.re_assign_label(raw_label1, label)  # re-assign the label according to the gt label
            acc_onet_tw, miou_onet_tw, dr_onet_tw, far_onet_tw, _ = uti.evaluate_nau_segmentation_v2(pred_label1, label)
            onettw_far_list.append(far_onet_tw)
            if torch.equal(raw_label1, pred_label1):  # if the label is not changed, Vd represents the foreground
                fg1 = Vd1
                bg1 = Vt1
            else:  # if the label is changed, Vt represents the foreground
                fg1 = Vt1
                bg1 = Vd1
            X2 = uti.tensor_normal_per_frame(fg1)
            Lt2, Vt2, Ld2, Vd2, S2 = onet2nd(X2)
            raw_label2 = onet2nd.predict_label(S2)  # 0 is foreground, 1 is background  in the case of normalized X'
            pred_label2 = uti.re_assign_label(raw_label2, label)  # re-assign the label according to the gt label
            acc_onet2, miou_onet2, dr_onet2, far_onet2, _ = uti.evaluate_nau_segmentation_v2(pred_label2, label)
            onet2_far_list.append(far_onet2)
            t_letters = ['d', 'f', 'i', 'j', 'k']
            #rmfont = {'fontname':'Times New Roman'}
            # // write your code related to basemap here
            # plt.title('title',**csfont)
            for i,name in enumerate(names): # index in one batch
                letter = name.split('_')[2]
                if letter in t_letters:
                    aid = t_letters.index(letter) # index in the figure
                    subimage = X1[i, 0, :, :].cpu().numpy()
                    gt_label  = label[i, :, :].cpu().numpy()
                    infoseg_pred = info_label[i, :, :].cpu().numpy()
                    onet_ws_pred = pred_label_ws[i, :, :].cpu().numpy()
                    onet_tw_pred = pred_label1[i, :, :].cpu().numpy()
                    onet2nd_pred = pred_label2[i, :, :].cpu().numpy()

                    axs[aid, 0].imshow(subimage)
                    axs[aid, 0].text(20, 30, letter.upper(), color='white', fontsize=16)
                    axs[aid, 1].imshow(gt_label)
                    axs[aid, 2].imshow(infoseg_pred)
                    axs[aid, 3].imshow(onet_tw_pred)
                    axs[aid, 4].imshow(onet_ws_pred)
                    axs[aid, 5].imshow(onet2nd_pred)
                    for j in range(6):
                        axs[aid, j].set_xticks([])
                        axs[aid, j].set_yticks([])
        axs[0, 2].set_title(r'Infoseg $P_{fa}=$%.4f' % (np.mean(infoseg_far_list)), fontsize=10)
        axs[0, 3].set_title(r'Onet_TW $P_{fa}=$%.4f' % (np.mean(onettw_far_list)),  fontsize=10)
        axs[0, 4].set_title(r'Onet_WS $P_{fa}=$%.4f' % (np.mean(onetws_far_list)),  fontsize=10)
        axs[0, 5].set_title(r'Onet2   $P_{fa}=$%.4f' % (np.mean(onet2_far_list)),   fontsize=10)
        #print('\hline')
        #print('\multirow{2}{*}{%d}' % psnr)
        # print('CFAR  & %.4f & %.4f & %.4f & %.4f \\\\' % (acc_cfar, miou_cfar, dr_cfar, far_cfar))
        # print('Onet2 & %.4f & %.4f & %.4f & %.4f \\\\' % (acc1, miou1, dr1, far1))
        # plt.show()
        fig.savefig(os.path.join(config.out_root,'exp_naurain_infoseg_onettw_onetws_onet2_dfijk.png'), dpi=200, bbox_inches='tight')
        print('save the figure to %s' % os.path.join(config.out_root,'exp_naurain_infoseg_onettw_onetws_onet2_dfijk.png'))


def fig_cfar_infoseg_onet_ws_naurain_revision(onet_config, infoseg_config):
    '''
    Reviewer questioned that onet is not superior to infoseg. Only the ensemble of onet is better.
    Here in this figure, we plot infoseg, onet_tw, onet_ws and onet2(twin ensemble) for comparison.
    :param config:
    :return:
    '''
    infoseg = infoseg_model.Infoseg(input_channels=infoseg_config.input_chn, K=infoseg_config.gt_k,
                                    height=infoseg_config.input_sz, width=infoseg_config.input_sz)
    infoseg.to(infoseg_config.device)
    infoseg.load_state_dict(torch.load(infoseg_config.model_file,map_location=lambda storage, loc: storage)["net"])
    infoseg.eval()

    # onet1st = onet_simbg_model.Onet(binit=True)
    # onet1st.to(onet_config.device)
    # onet1st.load_state_dict(torch.load(onet_config.model_file_onet1st, map_location=lambda storage, loc: storage)['net'])
    #
    # onet2nd = onet_simbg_model.Onet(binit=True)
    # onet2nd.to(onet_config.device)
    # onet2nd.load_state_dict(torch.load(onet_config.model_file_onet2nd, map_location=lambda storage, loc: storage)['net'])
    # onet1st.eval()  # set the model to evaluation mode, without this line, the output will be different, more false alarms.
    # onet2nd.eval()

    onet_wtshare = onet_rev_model.Onet(binit=True, bshare=True)
    onet_wtshare.to(onet_config.device)
    onet_wtshare.load_state_dict(torch.load(onet_config.model_file, map_location=lambda storage, loc: storage)['net'])
    onet_wtshare.eval() # set the model to evaluation mode, without this line, the output will be different

    kval = 2.0  # far = 0.03
    cfar = cfar_model.CFAR(kval=kval, nref=16, mguide=8)

    fig, axs = plt.subplots(3, 5, figsize=(12, 12*3/5), dpi=200, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    plt.rcParams["font.serif"] = ["Times New Roman"]
    axs[0, 0].set_title('Input frame', fontsize=10)
    axs[0, 1].set_title('Ground truth',fontsize=10)
    # axs[0, 2].set_title('Infoseg',     fontsize=12)
    # axs[0, 3].set_title('Onet_TW',     fontsize=12)
    # axs[0, 4].set_title('Onet_WS',     fontsize=12)
    # axs[0, 5].set_title('Onet2',       fontsize=12)

    infoseg_far_list = []
    onettw_far_list  = []
    onetws_far_list  = []
    cfar_far_list    = []
    test_dataloader = naurain_model.make_nau_rain_dataloader(config)

    with torch.no_grad():
        for names, X1, label in test_dataloader: # batchsize=20, one batch contains 20 samples.

            label = label.to(config.device)
            X1 = X1.to(config.device)

            L, S, V = infoseg(X1)
            info_raw_label = infoseg.get_label(V)
            info_label = uti.re_assign_label(info_raw_label, label)  # re-assign the label according to the gt label
            acc_info, miou_info, dr_info, far_info, tiou_info = uti.evaluate_nau_segmentation_v2(info_label, label)
            infoseg_far_list.append(far_info)

            Lt_ws, Vt_ws, Ld_ws, Vd_ws, S_ws = onet_wtshare(X1)  # onet predict.
            raw_label_ws = onet_wtshare.predict_label(S_ws)  # 1 means Vd>Vt (fg is Vd), 0 means Vt>Vd (fg is Vt).
            pred_label_ws = uti.re_assign_label(raw_label_ws, label)  # re-assign the label according to the gt label
            acc_onet_ws, miou_onet_ws, dr_onet_ws, far_onet_ws, tiou_onet_ws = uti.evaluate_nau_segmentation_v2(pred_label_ws, label)
            onetws_far_list.append(far_onet_ws)



            t_letters = ['d', 'i', 'k']
            #rmfont = {'fontname':'Times New Roman'}
            # // write your code related to basemap here
            # plt.title('title',**csfont)
            for i,name in enumerate(names): # index in one batch
                letter = name.split('_')[2]
                if letter in t_letters:
                    aid = t_letters.index(letter) # index in the figure
                    subimage = X1[i, 0, :, :].cpu().numpy()
                    gt_label  = label[i, :, :].cpu().numpy()
                    infoseg_pred = info_label[i, :, :].cpu().numpy()
                    onet_ws_pred = pred_label_ws[i, :, :].cpu().numpy()

                    cfar_label = cfar.cfar_seg(subimage)
                    acc_cfar, miou_cfar, dr_cfar, far_cfar, tiou_cfar = uti.evaluate_nau_segmentation_v2(torch.tensor(cfar_label), torch.tensor(gt_label))
                    cfar_far_list.append(far_cfar)
                    axs[aid, 0].imshow(subimage)
                    axs[aid, 0].text(20, 30, letter.upper(), color='white', fontsize=16)
                    axs[aid, 1].imshow(gt_label)
                    axs[aid, 2].imshow(infoseg_pred)
                    axs[aid, 3].imshow(cfar_label)
                    axs[aid, 4].imshow(onet_ws_pred)
                    for j in range(5):
                        axs[aid, j].set_xticks([])
                        axs[aid, j].set_yticks([])
        axs[0, 2].set_title(r'Infoseg $P_{fa}=$%.4f' % (np.mean(infoseg_far_list)), fontsize=10)
        axs[0, 3].set_title(r'CFAR $P_{fa}=$%.4f' % (np.mean(cfar_far_list)),  fontsize=10)
        axs[0, 4].set_title(r'Onet $P_{fa}=$%.4f' % (np.mean(onetws_far_list)),  fontsize=10)
        #axs[0, 5].set_title(r'Onet2   $P_{fa}=$%.4f' % (np.mean(onet2_far_list)),   fontsize=10)
        #print('\hline')
        #print('\multirow{2}{*}{%d}' % psnr)
        # print('CFAR  & %.4f & %.4f & %.4f & %.4f \\\\' % (acc_cfar, miou_cfar, dr_cfar, far_cfar))
        # print('Onet2 & %.4f & %.4f & %.4f & %.4f \\\\' % (acc1, miou1, dr1, far1))
        # plt.show()
        fig.savefig(os.path.join(config.out_root,'exp_naurain_infoseg_cfar_onetws_dik.png'), dpi=200, bbox_inches='tight')
        print('save the figure to %s' % os.path.join(config.out_root,'exp_naurain_infoseg_cfar_onetws_dik.png'))
        #fig.savefig(os.path.join(config.out_root,'exp_naurain_infoseg_onettw_onetws_onet2_dfijk.png'), dpi=200, bbox_inches='tight')
        #print('save the figure to %s' % os.path.join(config.out_root,'exp_naurain_infoseg_onettw_onetws_onet2_dfijk.png'))

if __name__ == '__main__':

    datehour_mark = '%04d_%02d%02d_%02d' % (datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour)

    config = conf_model.generate_config('./configs/train_onet_20250407.yml', dataset_name='naurain')
    #infoseg_config = conf_model.generate_config('./configs/train_infoseg.yml', dataset_name='naurain')
    config.out_root = os.path.join(config.out_root, 'exp_naurain')
    print('checkpoint_directory:\n ', config.out_root)
    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    config.use_augmentation = config.aug

    print('Onet_pid: ', os.getpid())
    print('current w_dir ', os.getcwd())

    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    logging.info(uti.config_to_str(config))  # print configuration to log file.
    logging.info('Onet_pid: %d' % os.getpid())
    logging.info('current w_dir %s' % os.getcwd())

    print(uti.config_to_str(config))  # print configuration to log file.
    # nohup python -u Onet1.0_20230428.py > Onet1.0_train_on_psnr0_test_on_psnr0-2.log &

    onet = onet_rev_model.Onet(in_chns=1, binit=False, bshare=True)  # init onet_low_snr at the second time.
    onet.to(config.device)
    #config.model_name = 'onet_tau'
    #config.model_file = ('/root/tip_onet2.0_202403/checkpoint/onet1.0_renew/weight_share/onet_weight_share_PSNR0-2_epoch_500_2024_0503_08.pytorch')
    model_file = os.path.join(config.out_root, config.model_file)
    onet.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['net'])
    #fig_cfar_infoseg_onet_ws_naurain_revision(config, infoseg_config)
    #exit(0)
    # fig_onet2_stage_vs_onet_weight_share_naurain_revision(config)
    # exit(0)
    #fig_cfar_vs_onet_weight_share_naurain_revision(config)
    #exit(0)
    # fig_onet_weightshare_naurain_islands_revision(config)
    # exit(0)

    # fig_infoseg_onet_twins_vs_onet_weight_share_naurain_revision(config, infoseg_config)
    # exit(0)
    # for table: Perf. in the rain clutter of marine radar.
    test_dataloader = naurain_model.make_nau_rain_dataloader(config)
    acc, miou, pd, far, psnr, snr =test_naurain_onet(config, onet, test_dataloader)
    logging.info('acc: %.4f, miou: %.4f, pd: %.4f, far: %.4f, psnr:%.4f, snr:%.4f ' % (acc, miou, pd, far, psnr, snr))
    print('naurain results acc: %.4f, miou: %.4f, pd: %.4f, far: %.4f, psnr:%.4f, snr:%.4f ' % (acc, miou, pd, far, psnr, snr))




