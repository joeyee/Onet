'''
Using the weight-share and dot-product to train the Onet model. 

Created by ZhouYi@Linghai_Dalian on 2025/4/7.

Reference code files:
[1] 'Onet_v1_20230428.py'.
[2]  'Onet_L4H4_Dot_vanilla_20240503.py
[3] 'Train_Onet_on_simclutter_20240606.py'
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
#import evaluation_20230429               as eval_model  # load evaluation function.
import dataloader.simbg4onet_20230209     as simbg_model # load simbg_datasets.
import Onet_vanilla_20240606              as onet_vanilla_model

# for ablation
# import Onet_vanilla_outc_20240609         as onet_vanilla_outc_model
# import Onet_ConvNeXt_outc_20240609        as onet_convnext_outc_model
# import Onet_Trans_Unet_20240613           as onet_transunet_model
# import Onet_Swin_config_20230408          as onet_swin_config
# import Onet_Swin_Transformer_20240615     as onet_swin_model
# import onet_ablation_RSN_20230626         as onet_vanilla_rsn_model # for ablataion on random sampling negative
import pandas as pd
import gc
import glob
import logging

torch.manual_seed(1981)
np.random.seed(1981)
torch.set_default_dtype(torch.float32)
def measure_snr_on_fg(config,  onet, test_loader):
    '''
    Measure psnr(peak point), snr(target region) on segmentated foreground.
    :param config:
    :param onet:
    :param test_loader:
    :return:
    '''
    onet.eval()
    input_psnr_list = []
    input_snr_list =  []
    fg_psnr_list =  []
    fg_snr_list  =  []
    with torch.no_grad():
        for X, label, psnr in test_loader:
            X = X.to(config.device)
            label = label.to(config.device)
            Lt, Vt, Ld, Vd, S = onet(X)

            St = S[:, 0, :, :]
            Sd = S[:, 1, :, :]

            Vt = uti.tensor_normal_per_frame(Vt)
            Vd = uti.tensor_normal_per_frame(Vd)

            # pred_label = 1 - onet.predict_label(S) # 1 is foreground, 0 is background  in the case of un_normalize X
            raw_label = onet.predict_label(S)  # 0 is foreground, 1 is background  in the case of strong foreground (e.g. fg is cloud).
            pred_label = uti.re_assign_label(raw_label, label)  # re-assign the label to make sure the foreground is 1
            pred_t = Vt.squeeze(dim=1)
            pred_d = Vd.squeeze(dim=1)

            ideal_psnr = psnr[0].item()
            assert(torch.all(psnr == ideal_psnr)) # 'psnr equal each batch

            if torch.equal(raw_label,pred_label):  # if the label is not changed, Vd represents the foreground
                fg = pred_d
            else:  # if the label is changed, Vt represents the foreground
                fg = pred_t
            input_psnr, input_snr  = uti.get_psnr(X.squeeze(dim=1), label) # core function
            fg_psnr, fg_snr=uti.get_psnr(fg, label) # core function
            fg_psnr_list.append(fg_psnr)
            fg_snr_list.append(fg_snr)
            input_psnr_list.append(input_psnr)
            input_snr_list.append(input_snr)

    fg_psnr = np.array(fg_psnr_list).mean()
    fg_snr  = np.array(fg_snr_list).mean()
    input_snr = np.array(input_snr_list).mean()
    input_psnr = np.array(input_psnr_list).mean()
    return input_psnr, input_snr, fg_psnr, fg_snr


def test_simclutter(str_txt, config, onet, test_loader, verbose=1, measure_snr=False):
    onet.eval()
    acc_batch_list = []
    miou_batch_list = []
    dr_batch_list = []
    far_batch_list = []
    tiou_batch_list = [] #contains the target iou of the predcition
    # real_psnr_list = []
    # real_snr_list  = []
    random_batch = np.random.randint(0, len(test_loader))
    bid = 0
    with torch.no_grad():
        for X, label, psnr in test_loader:
            X = X.to(config.device)

            label = label.to(config.device)
            Lt, Vt, Ld, Vd, S = onet(X)

            St = S[:, 0, :, :]
            Sd = S[:, 1, :, :]

            Vt = uti.tensor_normal_per_frame(Vt)
            Vd = uti.tensor_normal_per_frame(Vd)
            # S = torch.softmax(torch.concat([Vt, Vd], dim=1), dim=1)
            # pred_label = onet.predict_label(S)
            #pred_label = 1 - onet.predict_label(S) # 1 is foreground, 0 is background  in the case of un_normalize X
            raw_label = onet.predict_label(S)  # 0 is foreground, 1 is background  in the case of strong foreground (e.g. fg is cloud).
            pred_label= raw_label
            # for n in range(raw_label.shape[0]):
            #     pred_label[n,::] = uti.re_assign_label(raw_label[n,::], label[n,::])
            pred_label = uti.re_assign_label(raw_label, label)  # re-assign the label to make sure the foreground is 1
            pred_t = Vt.squeeze(dim=1)
            pred_d = Vd.squeeze(dim=1)

            # if measure_snr:  # in this flag, psnr is equal each path.
            #     assert(torch.all(psnr == psnr[0].item())) # 'psnr equal each batch
            #
            #     if torch.equal(raw_label,pred_label):  # if the label is not changed, Vd represents the foreground
            #         fg = pred_d
            #     else:  # if the label is changed, Vt represents the foreground
            #         fg = pred_t
            #     real_psnr, real_snr=uti.get_psnr(fg, label)
            #     real_psnr_list.append(real_psnr)
            #     real_snr_list.append(real_snr)
            batch_acc, batch_miou, batch_dr, batch_far, batch_tiou = uti.evaluate_nau_segmentation_v2(pred_label, label)
            acc_batch_list.append(batch_acc)
            miou_batch_list.append(batch_miou)
            dr_batch_list.append(batch_dr)
            far_batch_list.append(batch_far)
            tiou_batch_list.append(batch_tiou)
            if bid == random_batch and verbose: #show random batch
                watch_batch = {'X':X, 'pred_t':pred_t, 'pred_d':pred_d, 'label':label, 'pred_label':pred_label, 'psnr':psnr,
                               'St':St, 'Sd':Sd, 'bid': bid}
                if verbose == 2:
                    torch.save(watch_batch, os.path.join(config.out_root, "watch_batch_psnr%d.pytorch" % psnr[0]))
            bid += 1
        #
    acc  = np.array(acc_batch_list).mean()     # overall accuracy
    miou = np.array(miou_batch_list).mean()    # mean iou
    dr   = np.array(dr_batch_list).mean()      # detection rate
    far  = np.array(far_batch_list).mean()     # false alarm rate
    tiou = np.array(tiou_batch_list).mean()    # target iou
    # if measure_snr:
    #     real_psnr = np.array(real_psnr_list).mean()
    #     real_snr  = np.array(real_snr_list).mean()
    #     print('real_psnr: %.4f, real_snr: %.4f' % (real_psnr, real_snr))
    if verbose:
        five_snrs = watch_batch['psnr'].tolist()[0:5]
        snr_txt = '_'.join(str(s) for s in five_snrs)
        txt = '%s_bid_%04d_acc_%.4f_miou_%.4f_dr_%.4f_far_%.4f_tiou_%.4f_snr%s' \
              % (str_txt, random_batch, acc, miou, dr, far, tiou, snr_txt)
        uti.show_unet_adversarial_v2(watch_batch['X'], watch_batch['pred_t'], watch_batch['pred_d'],
                                     watch_batch['label'], watch_batch['pred_label'], txt, config)

    return acc, miou, dr, far, tiou

def unsupervised_training_simclutter(config, onet, train_loader, test_loader):
    '''
    unsupervised training of the Onet directly on the test_set without any annotation.
    :return:
    '''
    print('Start training %s in %d epoches' % (config.model_name, config.epoch_nums))
    # for training...
    opt = torch.optim.Adam(onet.parameters(), lr=1e-5 / 2, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0, amsgrad=False)  # maximize=False)

    loss_list = []  # training record for all epoches
    jsd_list  = []

    acc_list  = []
    miou_list = []
    dr_list   = []
    far_list  = []
    tiou_list = [] # target iou

    best_loss = 1e6
    best_jsd = -100
    best_far = 1e6
    best_acc = 0
    best_dr  = 0
    tr_fig, axs = plt.subplots(1,2,figsize=(8, 8))  # show train loss
    # fig, ax = plt.subplots(1,2)

    for epoch in range(config.epoch_nums):
        config.last_epoch = epoch
        onet.train()
        loss_batch_list = []
        jsd_batch_list = []
        jsd_batch_dict = {'epoch': [epoch], 'jsd_top': [], 'jsd_dwn': [], 'jsd': [], 'loss': []}
        # random_batch = np.random.randint(0, len(test_loader))
        # bid =0
        for X, labels, snrs in train_loader:  # train_labels is not use.
            onet.zero_grad()
            X = X.to(config.device)

            Lt, Vt, Ld, Vd, S = onet(X)  # send rgb to onet for convnext backbone
            St = S[:, 0, :, :].unsqueeze(dim=1)
            Sd = S[:, 1, :, :].unsqueeze(dim=1)
            loss = onet.compute_loss(Lt, St, Ld, Sd)
            loss.backward()
            opt.step()
            jsd_batch_dict['loss'].append(loss.item())
            # uti.print_parameters_statics(onet)
            # uti.compare_top_dwn_parameters_statics(onet.topu, onet.dwnu)
            loss_batch_list.append(loss.item())
        # saving statistics in each epoch.
        loss_epoch = np.mean(np.array(loss_batch_list))
        loss_list.append(loss_epoch)


        str_txt = 'epoch_%04d_loss_%.4f' % (epoch, loss_epoch)
        if epoch % 50 == 0:
            acc_epoch, miou_epoch, dr_epoch, far_epoch, tiou_epoch = test_simclutter(str_txt, config, onet,
                                                                                                  test_loader,
                                                                                                  verbose=1)
        # else:
        #     acc_epoch, miou_epoch, dr_epoch, far_epoch, tiou_epoch = test_simclutter(str_txt, config, onet,
        #                                                                                           test_loader,
        #                                                                                           verbose=0)
            acc_list.append(acc_epoch)
            miou_list.append(miou_epoch)
            dr_list.append(dr_epoch)
            far_list.append(far_epoch)
            tiou_list.append(tiou_epoch)
            print("%s===Epoch: %04d loss: %.5f, lr: %.10f, acc:%.4f, miou:%.4f, target_iou:%.4f, dr:%.4f, far:%.2E, %s" %
                  (config.model_name, epoch, loss_epoch, opt.param_groups[0]['lr'],
                   acc_epoch, miou_epoch, tiou_epoch, dr_epoch, far_epoch, datetime.now()))
            logging.info("%s===Epoch: %04d loss: %.5f, lr: %.10f, acc:%.4f, miou:%.4f, target_iou:%.4f, dr:%.4f, far:%.2E, %s" %
                    (config.model_name, epoch, loss_epoch, opt.param_groups[0]['lr'],
                        acc_epoch, miou_epoch, tiou_epoch, dr_epoch, far_epoch, datetime.now()))
        if (epoch % 100 == 0 and epoch > 0):# decay the learning rate every 100 epoches
            opt.param_groups[0]['lr'] *= 0.5

        # if acc_epoch > best_acc:
        #     best_acc = acc_epoch
        #if best_loss > loss_epoch:
        #when dr_epoch is greater than 0.9, we only care about the far.
        if epoch == config.epoch_nums - 1 or epoch == 300:
        # if (dr_epoch  > best_dr and dr_epoch<0.90 and epoch > 5) or (dr_epoch>0.90 and far_epoch< best_far):# ignore the first 5 epoches where the model is not stable.
        #     best_far = far_epoch
        #     best_dr  = dr_epoch
        #     best_loss = loss_epoch
            datehour_mark = '%04d_%02d%02d_%02d' % (datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour)
            config.datehour_mark = datehour_mark
            print('Saving net dict at epoch %d' % epoch)
            logging.info('Saving net dict at epoch %d' % epoch)
            best_param_dict = onet.state_dict()
            save_dict_info  = {"net": onet.state_dict(), 'epoch': epoch}
            torch.save(save_dict_info, os.path.join(config.out_root, "%s_epoch_%d_%s.pytorch" % (config.model_name, epoch, datehour_mark)))
            #torch.save(save_dict_info, os.path.join(config.out_root, "Onet_%s_best_acc.pytorch" % (config.dataset)))

    # draw train loss
    tr_axs = axs[0]
    tr_axs.clear()
    tr_axs.plot(loss_list, 'r', label='train_loss')
    # tr_axs.plot(jsd_list,  'g-.',  label='jsd')
    tr_axs.legend()
    tr_fig.savefig(os.path.join(config.out_root, "Onet_simclutter_train_loss_%s.png" % datehour_mark))
    acc_axs = axs[1]
    acc_axs.clear()
    acc_axs.plot(acc_list, 'r', label='acc')
    acc_axs.plot(miou_list, 'g-.', label='miou')
    acc_axs.plot(dr_list, 'b--', label='dr')
    acc_axs.plot(far_list, 'k:', label='far')
    acc_axs.plot(tiou_list, 'm', label='tiou')
    acc_axs.legend()
    tr_fig.savefig(os.path.join(config.out_root, "Onet_simclutter_train_loss_%s.png" % datehour_mark))
    # del onet
    # torch.cuda.empty_cache()
    # gc.collect()
    #reload the best model
    #onet.load_state_dict(best_param_dict)
    print('Finish training Onet_simclutter at %s' % datetime.now())
    print('Best model saved at the epoch %d' % save_dict_info['epoch'])
    logging.info('Finish training Onet_simclutter at %s' % datetime.now())
    return onet


def test_2nd_stage_simclutter(str_txt, config, onet, onet2nd, test_loader, verbose=1):
    onet.eval()
    onet2nd.eval()
    acc_batch_list1 = []
    miou_batch_list1 = []
    dr_batch_list1 = []
    far_batch_list1 = []
    tiou_batch_list1 = []
    acc_batch_list2 = []
    miou_batch_list2 = []
    dr_batch_list2 = []
    far_batch_list2 = []
    tiou_batch_list2 = []
    random_batch = np.random.randint(0, len(test_loader))
    bid = 0
    with torch.no_grad():
        for X1, label, snr in test_loader:
            label = label.to(config.device)
            X1 = X1.to(config.device)
            Lt1, Vt1, Ld1, Vd1, S1 = onet(X1)  # onet predict.
            pred_t1 = Vt1.squeeze(dim=1)
            pred_d1 = Vd1.squeeze(dim=1)
            raw_label1 = onet.predict_label(S1)  # 0 is foreground, 1 is background  in the case of normalized X'
            pred_label1 = uti.re_assign_label(raw_label1, label)  # re-assign the label according to the gt label
            batch_acc1, batch_miou1, batch_dr1, batch_far1, batch_tiou1 = uti.evaluate_nau_segmentation_v2(pred_label1, label)
            acc_batch_list1.append(batch_acc1)
            miou_batch_list1.append(batch_miou1)
            dr_batch_list1.append(batch_dr1)
            far_batch_list1.append(batch_far1)
            tiou_batch_list1.append(batch_tiou1)

            if torch.equal(raw_label1,pred_label1):  # if the label is not changed, Vd represents the foreground
                X2 = Vd1
            else:  # if the label is changed, Vt represents the foreground
                X2 = Vt1

            X2 = uti.tensor_normal_per_frame(X2)
            Lt2, Vt2, Ld2, Vd2, S2 = onet2nd(X2)
            pred_t2 = Vt2.squeeze(dim=1)
            pred_d2 = Vd2.squeeze(dim=1)
            raw_label2  = onet2nd.predict_label(S2)  # 0 is foreground, 1 is background  in the case of normalized X'
            pred_label2 = uti.re_assign_label(raw_label2, label)  # re-assign the label according to the gt label
            if torch.equal(raw_label2,pred_label2):  # if the label is not changed, Vd represents the foreground
                fg = Vd2
                bg = Vt2
            else:
                fg = Vt2
                bg = Vd2

            batch_acc, batch_miou, batch_dr, batch_far, batch_tiou = uti.evaluate_nau_segmentation_v2(pred_label2, label)
            acc_batch_list2.append(batch_acc)
            miou_batch_list2.append(batch_miou)
            dr_batch_list2.append(batch_dr)
            far_batch_list2.append(batch_far)
            tiou_batch_list2.append(batch_tiou)
            if bid == random_batch and verbose: #show random batch
                watch_batch1 = {'X':X1, 'pred_t':pred_t1, 'pred_d':pred_d1, 'label':label, 'pred_label':pred_label1, 'psnr':snr,'bid': bid}
                watch_batch2 = {'X':X2, 'pred_t':pred_t2, 'pred_d':pred_d2, 'label': label, 'pred_label': pred_label2, 'psnr': snr, 'bid': bid}
                if verbose == 2:
                    five_snrs = watch_batch1['psnr'].tolist()[0:5]
                    snr_txt = '_'.join(str(s) for s in five_snrs)
                    torch.save(watch_batch1, os.path.join(config.out_root, "watch_batch1_snr%s.pytorch" % snr_txt))
                    torch.save(watch_batch2, os.path.join(config.out_root, "watch_batch2_snr%s.pytorch" % snr_txt))
            bid += 1
        #
    #Performance in the first stage
    acc1  = np.array(acc_batch_list1).mean()     # overall accuracy
    miou1 = np.array(miou_batch_list1).mean()    # mean iou
    dr1   = np.array(dr_batch_list1).mean()      # detection rate
    far1  = np.array(far_batch_list1).mean()     # false alarm rate
    tiou1 = np.array(tiou_batch_list1).mean()    # target iou
    #Performance in the second stage
    acc2  = np.array(acc_batch_list2).mean()     # overall accuracy
    miou2 = np.array(miou_batch_list2).mean()    # mean iou
    dr2   = np.array(dr_batch_list2).mean()      # detection rate
    far2  = np.array(far_batch_list2).mean()     # false alarm rate
    tiou2 = np.array(tiou_batch_list2).mean()    # target iou
    five_snrs = snr.tolist()[0:5]
    snr_txt = '_'.join(str(s) for s in five_snrs)
    if verbose:
        five_snrs = watch_batch1['psnr'].tolist()[0:5]
        snr_txt = '_'.join(str(s) for s in five_snrs)
        txt = '%s_onet1st_bid_%04d_acc_%.4f_miou_%.4f_dr_%.4f_far_%.4f_snr%s' \
              % (str_txt, random_batch, acc1, miou1, dr1, far1, snr_txt)
        # X, pred_t, pred_d, label, pred_label # input order
        # X, label,  pred_label, pred_t, pred_d # show image order
        uti.show_unet_adversarial_v2(watch_batch1['X'], watch_batch1['pred_t'], watch_batch1['pred_d'],
                                     watch_batch1['label'], watch_batch1['pred_label'], txt, config)
        txt = '%s_onet2nd_bid_%04d_acc_%.4f_miou_%.4f_dr_%.4f_far_%.4f_snr%s' \
              % (str_txt, random_batch, acc2, miou2, dr2, far2, snr_txt)
        uti.show_unet_adversarial_v2(watch_batch2['X'], watch_batch2['pred_t'], watch_batch2['pred_d'],
                                     watch_batch2['label'], watch_batch2['pred_label'], txt, config)
        print('snr:%s, acc1:%.4f, miou1:%.4f, tiou:%.4f, dr1:%.4f, far1:%.4f' % (snr_txt, acc1, miou1, tiou1, dr1, far1))
        print('snr:%s, acc2:%.4f, miou2:%.4f, tiou:%.4f, dr2:%.4f, far2:%.4f' % (snr_txt, acc2, miou2, tiou2, dr2, far2))
    return acc2, miou2, dr2, far2, tiou1



def verify_2nd_stage_onet(config, onet, onet2nd, verbose=1):
    #verbose=1 save sample figures.
    #verbose=2 will save batch results.
    psnrs = np.arange(0,11)
    performance_dict = {}
    for psnr in psnrs:
        # if snr!=3:
        #     continue
        data_loader = simbg_model.make_dataloader_via_snr(config, psnr) # make dataloader for each snr
        str_txt = 'verify_psnr_%d' % psnr
        # verbose=2 will save the prediction results in batch.
        acc, miou, dr, far, tiou= test_2nd_stage_simclutter(str_txt, config, onet, onet2nd, data_loader, verbose) # get the performance of onet  in different snr.
        performance_dict[psnr] = {'acc': acc, 'miou': miou, 'tiou': tiou, 'dr': dr, 'far': far}
        logging.info('psnr:%02d, acc:%.4f, miou:%.4f, tiou:%.4f, dr:%.4f, far:%.4f' % (psnr, acc, miou, tiou, dr, far))
    performance_dict['ave'] = {'acc': np.mean([performance_dict[psnr]['acc'] for psnr in psnrs]),
                               'miou': np.mean([performance_dict[psnr]['miou'] for psnr in psnrs]),
                               'tiou': np.mean([performance_dict[psnr]['tiou'] for psnr in psnrs]),
                               'dr': np.mean([performance_dict[psnr]['dr'] for psnr in psnrs]),
                               'far': np.mean([performance_dict[psnr]['far'] for psnr in psnrs])}
    logging.info('PSNR0-10, ave_acc:%.4f, ave_miou:%.4f, ave_tiou:%.4f, ave_dr:%.4f, ave_far:%.4f' % (performance_dict['ave']['acc'],
                                                                                                      performance_dict['ave']['miou'],
                                                                                                      performance_dict['ave']['tiou'],
                                                                                                      performance_dict['ave']['dr'],
                                                                                                      performance_dict['ave']['far']))
    return performance_dict

def verify_onet_simclutter(config, onet, verbose=1):
    '''
    get the performance of onet on the test set of simclutter in different snr.
    :param config:
    :param onet:
    :return:
    '''
    #psnrs = [0, 2, 4, 5, 6, 8, 10]
    psnrs = np.arange(0,11)
    acc_list  = []
    miou_list = []
    pd_list   = []
    far_list  = []
    tiou_list = []
    performance_dict = {}
    for psnr in psnrs:
        data_loader = simbg_model.make_dataloader_via_snr(config, psnr) # make dataloader for each snr
        str_txt = '%s_verify_snr_%d' % (config.dataset, psnr)
        #verbose=2 will save the prediction results in batch.
        #verbose=1 will save image samples
        acc, miou, pd, far, tiou = test_simclutter(str_txt, config, onet, data_loader, verbose=False) # get the performance of onet  in different snr.
        input_psnr, input_snr, fg_psnr, fg_snr= measure_snr_on_fg(config, onet, data_loader)
        print('psnr:%02d, acc:%.4f, miou:%.4f, tiou:%.4f, pd:%.4f, far:%.4f, input_psnr:%.2f, input_snr:%.2f, fg_psnr:%.2f, fg_snr:%.2f'
              % (psnr, acc, miou, tiou, pd, far, input_psnr, input_snr, fg_psnr, fg_snr))
        performance_dict[psnr] = {'acc': acc, 'miou': miou, 'tiou': tiou, 'pd': pd, 'far': far}
        acc_list.append(acc)
        miou_list.append(miou)
        pd_list.append(pd)
        far_list.append(far)
        tiou_list.append(tiou)
    print('PSNR0-10, ave_acc:%.4f, ave_miou:%.4f, ave_tiou:%.4f, ave_pd:%.4f, ave_far:%.4f' %
          (np.mean(acc_list), np.mean(miou_list), np.mean(tiou_list), np.mean(pd_list), np.mean(far_list)))
    performance_dict['ave'] = {'acc': np.mean(acc_list), 'miou': np.mean(miou_list), 'tiou': np.mean(tiou_list),
                                     'pd': np.mean(pd_list), 'far': np.mean(far_list)}
    return performance_dict

def train_onet_by_snr(config, onet):
    '''
    Train onet snr by snr. And Test it's performance on the whole test set (including all snrs).
    :param config:
    :return:
    '''
    #onet = Onet()  # no bias.
    onet.to(config.device)
    init_param_dict = onet.state_dict()
    for psnr in range(0,11):
    #snr = 0
        print('=== train onet at psnr %d ===' % psnr)
        config.out_root = os.path.join('/home/ubuntu/tip2022/checkpoint/sim_clutter/', 'onet_snr_%02d'%psnr)
        if not os.path.exists(config.out_root):
            os.makedirs(config.out_root)
        data_loader = simbg_model.make_dataloader_via_snr(config, psnr)
        # onet = Onet()   # no bias.
        # onet.to(config.device)
        onet.load_state_dict(init_param_dict) # make sure the init param is the same for each snr.
        onet = unsupervised_training_simclutter(config, onet, data_loader, data_loader) # return onet with the best dr
        verify_onet_simclutter(config, onet)
        # del onet
        # torch.cuda.empty_cache()
        # gc.collect()
def test_onet_by_snr(config, onet):
    '''
    :param config:
    :return:
    '''
    #snr = 0
    # config.out_root = os.path.join('/home/ubuntu/tip2022/checkpoint/sim_clutter/', 'onet_snr_%02d' % snr)
    # if not os.path.exists(config.out_root):
    #     os.makedirs(config.out_root)
    # data_loader = simbg_model.make_dataloader_via_snr(config, snr)
    #onet = Onet()  # no bias. ablation, binit=True
    onet.to(config.device)
    if config.restart:
        onet.load_state_dict(torch.load(config.model_file, map_location=lambda storage, loc: storage)['net'])
    psnrs = np.arange(0,11)
    performance_dict = {}
    for psnr in psnrs:
        data_loader = simbg_model.make_dataloader_via_snr(config, psnr) # make dataloader for each psnr
        str_txt = 'psnr_%d' % psnr
        acc, miou, dr, far, tiou = test_simclutter(str_txt, config, onet, data_loader, verbose=1, twice_flag=False) # get the performance of onet  in different snr.
        print('psnr:%02d, acc:%.4f, miou:%.4f, tiou:%.4f, dr:%.4f, far:%.4f' % (psnr, acc, miou, tiou, dr, far))
        performance_dict[psnr] = {'acc': acc, 'miou': miou, 'tiou': tiou, 'dr': dr, 'far': far}
    #compute the average acc, miou, tiou, dr, far
    ave_acc = np.mean([performance_dict[psnr]['acc'] for psnr in psnrs])
    ave_miou = np.mean([performance_dict[psnr]['miou'] for psnr in psnrs])
    ave_tiou = np.mean([performance_dict[psnr]['tiou'] for psnr in psnrs])
    ave_dr = np.mean([performance_dict[psnr]['dr'] for psnr in psnrs])
    ave_far = np.mean([performance_dict[psnr]['far'] for psnr in psnrs])
    print('PSNR0-10, ave_acc:%.4f, ave_miou:%.4f, ave_tiou:%.4f, ave_dr:%.4f, ave_far:%.4f' % (ave_acc, ave_miou, ave_tiou, ave_dr, ave_far))
    performance_dict['0-10 Ave.']={'acc': ave_acc, 'miou': ave_miou, 'tiou': ave_tiou, 'dr': ave_dr, 'far': ave_far}
    return performance_dict

def test_model_performance(config, onet):
    '''
    :param config:
    :return:
    '''
    #onet = Onet()  # no bias.
    onet.to(config.device)
    model_files = glob.glob(os.path.join(config.model_root, '*.pytorch'))
    model_files.sort()
    for model_file in model_files:
        print('=== test model %s ===' % model_file)
        onet.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['net'])
        verify_onet_simclutter(config, onet, verbose=0)  # verbose=0 not save the prediction results and images in batch.

# def test_two_stage_onet_in_each_snr(config):
#     onet1st = onet_vanilla_model.Onet(binit=True)
#     onet1st.to(config.device)
#     onet1st.load_state_dict(torch.load(config.model_file_onet1st, map_location=lambda storage, loc: storage)['net'])
#     verify_onet_simclutter(config, onet1st, verbose=1) # verify the onet1st performance and check the snr improvement.
#
#     onet2nd = onet_vanilla_model.Onet(binit=True)
#     onet2nd.to(config.device)
#     onet2nd.load_state_dict(torch.load(config.model_file_onet2nd, map_location=lambda storage, loc: storage)['net'])
#     verify_2nd_stage_onet(config, onet1st, onet2nd, verbose=1) # verify the two onets performance on 20230515.

def test_model_performance(config):
    '''
    :param config:
    :return:
    '''
    onet = onet_vanilla_model.Onet(in_chns=1, binit=True, bshare=config.weight_share) # init onet_low_snr at the second time.
    onet.to(config.device)
    model_files = glob.glob(os.path.join(config.out_root, '*.pytorch'))
    model_files.sort()
    for model_file in model_files:
        print('=== test model %s ===' % model_file)
        onet.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)['net'])
        verify_onet_simclutter(config, onet, verbose=0)  # verbose=0 not save the prediction results and images in batch.

def train_vanilla_rsn_on_simclutter():
    '''
    Train the vanilla onet by random sampling negative RSN samples on the simclutter dataset.
    :param config:
    :return:
    '''
    import pytz
    config = conf_model.generate_config('./configs/train_onet_vallina_20240606.yml', dataset_name='Rayleigh')
    config.out_root = '/root/tip_onet2.0_202403/checkpoint/sim_clutter/onet_vanilla_rsn'
    config.use_augmentation = config.aug
    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    current_date = datetime.now(pytz.timezone('Asia/Shanghai'))  # change time zone to Beijing
    datehour_mark = '%04d%02d%02d_%02d' % (current_date.year, current_date.month, current_date.day, current_date.hour)
    config.datehour_mark = datehour_mark
    print('Onet_pid: ', os.getpid())
    print('current w_dir ', os.getcwd())
    print('checkpoint_directory:\n ', config.out_root)
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

    bshare = True

    onet_low_snr = onet_vanilla_rsn_model.Onet(in_chns=config.input_chn, bshare=bshare)  # using twin modes
    onet_low_snr.to(config.device)
    config.dataset = 'simclutter_snr0-2'  # name prefix for the model file.
    if bshare == True:
        config.model_name = 'onet_weight_share_PSNR0-2'
    else:
        config.model_name = 'onet_twins_PSNR0-2'
    print(uti.count_parameters(onet_low_snr))
    train_lowsnr_dataloader, test_lowsnr_dataloader = simbg_model.make_dataloader_snr_range(config, low_snr=0, high_snr=2)  # snr ==1
    onet_low_snr = unsupervised_training_simclutter(config, onet_low_snr, train_lowsnr_dataloader,test_lowsnr_dataloader)


    config.dataset = 'simclutter_snr5-10'  # name prefix for the model file.
    if bshare == True:
        config.model_name = 'onet_weight_share_PSNR5-10'
    else:
        config.model_name = 'onet_twins_PSNR5-10'
    # config.model_name = 'onet_weight_share_PSNR5-10'
    train_highsnr_dataloader, test_highsnr_dataloader = simbg_model.make_dataloader_snr_range(config, low_snr=5, high_snr=10)  # 5 <= snr <= 10
    onet_high_snr = onet_vanilla_rsn_model.Onet(in_chns=config.input_chn, bshare=bshare)  # using twin modes
    onet_high_snr.to(config.device)
    onet_high_snr = unsupervised_training_simclutter(config, onet_high_snr, train_highsnr_dataloader, test_highsnr_dataloader)
    verify_onet_simclutter(config, onet_high_snr)
    verify_2nd_stage_onet(config, onet_low_snr, onet_high_snr, verbose=1)


# nohup python -u Train_Onet_vanilla_on_simclutter_20240606.py > Train_Onet_vanilla_on_simclutter_20240606.out &
# nohup python -u Train_Onet_vanilla_on_simclutter_20240606.py > Train_Onet_vanilla_outc_on_simclutter_20240609.out &
# nohup python -u Train_Onet_vanilla_on_simclutter_20240606.py > Train_Onet_vanilla_outc_tw_on_simclutter_20240610.out &

# nohup python -u Train_Onet_vanilla_on_simclutter_20240606.py > Train_Onet_convnext_outc_ws_on_simclutter_20240610.out &  # test on guangzhou server.
# nohup python -u Train_Onet_vanilla_on_simclutter_20240606.py > Train_Onet_convnext_outc_tw_on_simclutter_20240611.out &  #
# nohup python -u Train_Onet_on_simclutter_20240606.py > Train_Onet_transunet_on_simclutter_20240613.out &  #
# nohup python -u Train_Onet_on_simclutter_20240606.py > Train_Onet_swin_on_simclutter_20240615.out &
# nohup python -u Train_Onet_on_simclutter_20240606.py > ./checkpoint/sim_clutter/onet_vanilla_rsn/Train_Onet_vanilla_ws_rsn_on_simclutter_20240618.out &
# nohup python -u Train_Onet_on_simclutter_20250407.py > ./checkpoint/sim_clutter/onet_vallina/Train_Onet_vallina_on_simclutter_20250407.out &
if __name__ == '__main__':

    datehour_mark = '%04d_%02d%02d_%02d' % (datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour)
    #config = conf_model.generate_config('./configs/train_onet_vallina_20240606.yml', dataset_name='Rayleigh')
    #config = conf_model.generate_config('./configs/train_onet_convnext_20240607.yml', dataset_name='Rayleigh')
    #config = conf_model.generate_config('./configs/train_onet_transunet_20240613.yml', dataset_name='Rayleigh')
    #config = conf_model.generate_config('configs/train_onet_swin_20240615.yml', dataset_name='Rayleigh')
    #config.out_root = os.path.join(config.out_root, 'outc_weight_share')
    #config.out_root = os.path.join(config.out_root, 'outc_twins')
    config = conf_model.generate_config('configs/train_onet_20250407.yml', dataset_name='Rayleigh')
    print('checkpoint_directory:\n ', config.out_root)
    if not os.path.exists(config.out_root):
        os.makedirs(config.out_root)
    config.use_augmentation = config.aug
    print('Onet_pid: ', os.getpid())
    print('current w_dir ', os.getcwd())

    test_model_performance(config)
    exit(0)

    # train onet in low_snr 0-2.
    log_file = os.path.join(config.out_root, "%s_%s.log" % (config.model_name, datehour_mark))
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    logging.info(uti.config_to_str(config))  # print configuration to log file.
    logging.info('Onet_pid: %d' % os.getpid())
    logging.info('current w_dir %s' % os.getcwd())
    print(uti.config_to_str(config))  # print configuration to log file.

    onet_low_snr = onet_vanilla_model.Onet(in_chns=1, binit=True, bshare=True)  # in revised tip, onet_vanilla use RGB for zy3.
    onet_low_snr.to(config.device)

    # print(uti.count_parameters(onet))
    # from thop import profile, clever_format
    # input = torch.randn(1, 1, 224, 224)
    # macs, params = profile(onet, inputs=(input.to(config.device),))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('Onet_vallina params:', params, ' macs: ', macs)
    # exit(0)
    #bshare = False # twin_model
    bshare = True # weight_share_model, False for twin_model

    if bshare == True:
        config.model_name = 'onet_weight_share_PSNR0-2'
    else:
        config.model_name = 'onet_twins_PSNR0-2'

    config.dataset = 'simclutter_snr0-2'  # name prefix for the model file.
    print(uti.count_parameters(onet_low_snr))
    # config.model_name = 'onet_weight_share_PSNR0-2'
    train_lowsnr_dataloader, test_lowsnr_dataloader = simbg_model.make_dataloader_snr_range(config, low_snr=0, high_snr=2)  # snr ==1
    onet_low_snr = unsupervised_training_simclutter(config, onet_low_snr, train_lowsnr_dataloader,test_lowsnr_dataloader)
    exit(0)

    #train onet in high_snr 5-10.
    onet_high_snr = onet_vanilla_model.Onet(in_chns=1, binit=True, bshare=True)  # in revised tip, onet_vanilla use RGB for zy3.
    onet_high_snr.to(config.device)
    config.dataset = 'simclutter_snr5-10'  # name prefix for the model file.
    if bshare == True:
        config.model_name = 'onet_weight_share_PSNR5-10'
    else:
        config.model_name = 'onet_twins_PSNR5-10'
    # config.model_name = 'onet_weight_share_PSNR5-10'
    train_highsnr_dataloader, test_highsnr_dataloader = simbg_model.make_dataloader_snr_range(config, low_snr=5, high_snr=10)  # 5 <= snr <= 10
    # train_highsnr_dataloader, test_highsnr_dataloader = simbg_model.make_dataloader_snr_range(config, low_snr=8, high_snr=10)  # 5 <= snr <= 10
    onet_high_snr = unsupervised_training_simclutter(config, onet_high_snr, train_highsnr_dataloader, test_highsnr_dataloader)
    verify_onet_simclutter(config, onet_high_snr)

    exit(0)
