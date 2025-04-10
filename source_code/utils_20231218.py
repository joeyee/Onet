'''
utility function support the unsupervised binary semantic segmentation
( such as Onet, infoseg, iic) architecture in training, testing.

Based on the source of utils.py of tip2022[Tencent_cloud_server].
Modified by Yi Zhou@Linghai_Dalian_2023-06-19 for 1st submission to TIP.
Updated by Yi Zhou@Linghai_Dalian_2023-12-18 for 1st revision.
'''
import torch
import numpy as np
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt

def config_to_str(config):
    '''uti_zy3_test_20240123.py
    import argparse
    parser = argparse.ArgumentParser()
    configs = parser.parse_args()  # get command line parameters
    :param config:
    :return:
    '''
    attrs = vars(config)
    string_val = "Config: -----\n"
    string_val += "\n".join("%s: %s" % item for item in attrs.items())
    string_val += "\n----------"
    return string_val

def count_parameters(net, factor=1, bverbose=True):
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params += params
        if bverbose:
            print(name, ' ', parameter.shape)
    # print(table)
    print(f"Total Trainable Params: {total_params*factor/1e6}M")
    cnt_str = "Total Trainable Params: %.2fM" % (total_params*factor/1e6)
    return cnt_str

def print_parameters_statics(net):
    # table = PrettyTable(["Modules", "Parameters"])
    with torch.no_grad():
        total_params = 0
        for name, parameter in net.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            # table.add_row([name, params])
            total_params += params
            #print(name, ' ', parameter.shape, ' mean %.5f'% torch.mean(parameter))
            if parameter.grad != None:
                print('%50s '%name,  '%32s'%str(parameter.shape), ' mean %.10f' % torch.mean(parameter),
                      'grad_mean %.10f'%torch.mean(parameter.grad))
            else:
                print('%50s '%name,  '%32s'%str(parameter.shape), ' mean %.10f' % torch.mean(parameter))
    return

def compare_top_dwn_parameters_statics(top_unet, dwn_unet):
    with torch.no_grad():
        total_params = 0
        for (top_name, top_parameter), (dwn_name, dwn_parameter) in zip(top_unet.named_parameters(), dwn_unet.named_parameters()):
            if not top_parameter.requires_grad: continue
            params = top_parameter.numel()
            # table.add_row([name, params])
            total_params += params*2
            #print(name, ' ', parameter.shape, ' mean %.5f'% torch.mean(parameter))
            if top_parameter.grad != None:
                print('top:%30s '%top_name,  '%32s'%str(top_parameter.shape), ' mean %.10f' % torch.mean(top_parameter),
                      'std %.10f' % torch.std(top_parameter),
                      'grad_mean %.10f'%torch.mean(top_parameter.grad), 'grad_std %.10f' % torch.std(top_parameter.grad))
                print('dwn:%30s '%dwn_name,  '%32s'%str(dwn_parameter.shape), ' mean %.10f' % torch.mean(dwn_parameter),
                      'std %.10f' % torch.std(dwn_parameter),
                      'grad_mean %.10f'%torch.mean(dwn_parameter.grad), 'grad_std %.10f' % torch.std(dwn_parameter.grad))
            else:
                print('top:%50s '%top_name, '%32s'%str(top_parameter.shape), ' mean %.10f' % torch.mean(top_parameter),
                      'std %.10f' % torch.std(top_parameter))
                print('dwn:%50s '%dwn_name, '%32s'%str(dwn_parameter.shape), ' mean %.10f' % torch.mean(dwn_parameter),
                      'std %.10f' % torch.std(dwn_parameter))
    return

def print_weights_and_grad(net):
    print("---------------")
    for n, p in net.named_parameters():
        print("%s abs: min %f max %f max grad %f" %
              (n, torch.abs(p.data).min(), torch.abs(p.data).max(), torch.abs(p.grad).max()))
    print("---------------")


def nice(d):
    res = ""
    for k in d.keys():
        v = d[k]
        res += ("\t%s: %s\n" % (k, v))
    return res


def _acc(preds, targets, num_k):
    '''
    pixel accuracy computing for evaluating the segmentation
    acc = (TP+TN)/(TP+TN+FP+FN)
    :param preds:
    :param targets:
    :param num_k:
    :return:
    '''
    assert (isinstance(preds, torch.Tensor) and
            isinstance(targets, torch.Tensor))

    assert (preds.shape == targets.shape)
    assert (preds.max() < num_k and targets.max() < num_k)


    acc = (preds == targets).sum().item() / float(preds.numel()) # acc = (TP+TN)/(TP+TN+FP+FN)
    return acc

def _miou(preds, targets, num_k):
    # The IoU score is calculated for each class separately and then averaged over
    # all classes to provide a global, mean IoU score of our semantic segmentation prediction.
    miou = 0.
    nums = 0.
    for k in range(num_k):
        gt_elem = (targets==k)
        pd_elem = (preds == k)
        if gt_elem.sum()==0 and pd_elem.sum()==0:
            miou += 1.0
            nums += 1
        elif gt_elem.sum()==0 and pd_elem.sum()!=0:
            #print('all gt are 0 for the class %d'%k)
            miou +=0
            nums +=1
            #continue # no gt for the class, not count the iou, but use the another class's iou.
        elif gt_elem.sum()!=0 and pd_elem.sum()==0:
            #print('all pd are 0 for the class %d'%k)
            #continue
            miou +=0
            nums +=1
        # if(gt_elem.sum()==0 or pd_elem.sum()==0): # gt or pd are pure background or foreground
        #     miou += (gt_elem == pd_elem).sum()/gt_elem.numel()
        # else: #mixed fg and bg
        else:
            intersection = torch.logical_and(gt_elem, pd_elem)
            union = torch.logical_or(gt_elem, pd_elem)
            miou += torch.sum(intersection) / torch.sum(union)
            nums +=1
    assert(nums==2 or nums==1)
    miou = miou.item()/nums # get mean float value, not tensor
    # assert num_k ==2
    # miou = 0.
    # cloud_iou = compute_iou(preds, targets)
    # back_iou  = compute_iou(1-preds, 1-targets)
    return miou

def _target_iou(preds, targets):
    '''
    compute the iou of the target region.
    :param preds:
    :param targets:
    :return:
    '''
    assert (isinstance(preds, torch.Tensor) and
            isinstance(targets, torch.Tensor))

    assert (preds.shape == targets.shape)
    assert (preds.max() < 2 and targets.max() < 2)

    intersection = torch.logical_and(targets, preds)
    union = torch.logical_or(targets, preds)
    iou = torch.sum(intersection) / (torch.sum(union)+np.spacing(1))
    return iou.item()

def _detection_rate(preds, targets):
    '''
    computing detection rate of the segmentation
    :param preds:    predicted labels
    :param targets:  gt labels
    :return:
    '''
    tp = torch.sum((targets==1)*(preds==1))  # true positive pixels
    gtp= torch.sum(targets==1) # gt positive pixels
    assert (gtp>=tp)
    dr = tp/(gtp+np.spacing(1)) # detection rate of the positive pixels
    return dr.item()

def _false_alarm_rate(preds, targets):
    fp = torch.sum((targets==0)*(preds==1)) # false positive pixels
    gtf= torch.sum(targets==0)              # gt negative pixels
    assert(gtf >= fp)
    far = fp/(gtf+np.spacing(1))
    return far.item()

def reorder_pred_label_v2(pred_label, echos):
    '''
    Suppose sea clutter region has the most intensity [label 1],
        background has the least [label 0]
    :param pred_label:
    :param echos:
    :return:
    '''
    assert (echos.numel() == pred_label.numel())
    pred_label = pred_label.view(pred_label.numel())  # vectorize the prediction and gt label
    echos = echos.reshape(echos.numel())

    reordered_preds = torch.zeros_like(pred_label)
    if  torch.mean(echos[pred_label==0]) > torch.mean(echos[pred_label==1]): # predict bg is more intensive, change the id.
        reordered_preds[pred_label==0] = 1
    else:
        reordered_preds[pred_label==1] = 1
    return reordered_preds

def evaluate_nau_segmentation_v2(predict_label, gt_label,  gt_k=2):
    '''
        nau_radar_datasets aims to supress the sea clutter in high sea state.
        Suppose sea clutter region has the most intensity [label 1],
        background has the least [label 0]
        :param predict_label:
        :param gt_label:
        :param echos: radar intensities
        :param gt_k:
        :return:
        '''
    assert (predict_label.numel() == gt_label.numel())
    predict_label = predict_label.view(predict_label.numel())  # vectorize the prediction and gt label
    gt_label      = gt_label.reshape(gt_label.numel())

    acc = _acc(predict_label, gt_label,  gt_k)  # omit the object's echos. acc and miou only for 0 and 1 labels.
    miou =_miou(predict_label, gt_label, gt_k) # note that only consider the miou of sea clutter and background
    dr  = _detection_rate(predict_label, gt_label)
    far = _false_alarm_rate(predict_label, gt_label)
    t_iou = _target_iou(predict_label, gt_label) #target iou

    return acc, miou, dr, far, t_iou

def get_psnr(img, label):
    '''
    :param img:  the source image
    :param label: the label on true target
    :return:
    '''
    #img = img.squeeze(dim=1)
    assert(img.shape == label.shape)
    assert(label.max() == 1 and label.min() == 0) # Binary label, 1 for targets, 0 for background.
    target = img*label
    target_pixels = torch.sum(label)
    peak   = torch.max(target)
    target_power = torch.sum(target ** 2) / target_pixels

    back  = img - target
    # Erc: average clutter energy.
    erc = torch.sum(back ** 2) / (img.numel()-target_pixels)

    psnr = 10 * torch.log10(peak ** 2 / erc)
    snr  = 10 * torch.log10(target_power / erc)
    return psnr.item(), snr.item()

def _hungarian_match(flat_preds, flat_targets, num_k):
    assert (isinstance(flat_preds, torch.Tensor) and
            isinstance(flat_targets, torch.Tensor))

    num_samples = flat_targets.shape[0]

    assert (flat_preds.numel() == flat_targets.numel())  # one to one
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_assignment(num_samples - num_correct)

    # return as list of tuples, out_c to gt_c
    res = []
    out_c = match[0]  # row index
    gt_c = match[1]  # column index
    # for out_c, gt_c in match:
    #     res.append((out_c, gt_c))
    for i in range(len(out_c)):
        res.append((out_c[i], gt_c[i]))

    return res

def reorder_pred_label(pred_label, echos, gt_k):
    '''
    reorder the pred_label according to the label region's echo intensity sum.
    the most refers to sea clutter, the least points to background.
    :param pred_label:
    :param echos:
    :param gt_k:
    :return:
    '''
    region_intensities = []
    for c in range(gt_k): #
        region_intensities.append(torch.sum(echos[pred_label==c]))
    int_sum = torch.tensor(region_intensities)

    _, rid = torch.sort(int_sum)  # sort the intensities in ascending order
    reordered_preds = torch.zeros_like(pred_label)
    # for i in range(gt_k):
    #     reordered_preds[pred_label == rid[i]] = i
    # note here bk=0, obj=2, sea_clutter=1
    reordered_preds[pred_label == rid[0]] = 0

    if gt_k==3:
        reordered_preds[pred_label == rid[2]] = 1
        reordered_preds[pred_label == rid[1]] = 2
    if gt_k==2:
        reordered_preds[pred_label == rid[1]] = 1
    return reordered_preds


def evaluate_nau_segmentation(predict_label, gt_label, echos, gt_k=3):
    '''
    nau_radar_datasets aims to supress the sea clutter in high sea state.
    Suppose sea clutter region has the most intensity [label 1],
    background has the least [label 0], median is other clutter [label 0]
    :param predict_label:
    :param gt_label:
    :param echos: radar intensities
    :param gt_k:
    :return:
    '''
    assert (2<=gt_k<=3) #[bk, sc] or [bk, sc, obj], at most 3 types: background, SeaClutter, object echos .
    assert(echos.numel()==gt_label.numel())
    assert(predict_label.numel() == gt_label.numel())
    predict_label = predict_label.view(predict_label.numel()) #vectorize the prediction and gt label
    gt_label      = gt_label.view(gt_label.numel())
    echos         = echos.reshape(echos.numel())
    num_samples   = predict_label.shape[0]

    region_intensities = []
    for c in range(gt_k): #
        region_intensities.append(torch.sum(echos[predict_label==c]))

    int_sum = torch.tensor(region_intensities)

    _, rid  = torch.sort(int_sum)  # sort the intensities in ascending order

    # background has the least intensity value, label 0
    # sea clutter has the greatest intensity, label 1
    #match = [(torch.argmin(int_sum), 0),  (torch.argmax(int_sum), 1)]
    match = [(rid[0], 0), (rid[-1], 1)] #, (rid[1], 2)] # (min_id,0_bk), (max_id, 1_sc), (mid_id, 2_obj)

    reordered_preds = torch.zeros_like(predict_label)
    # for i in range(gt_k):
    #     reordered_preds[predict_label == rid[i]] = i
    for pred_i, target_i in match:
        reordered_preds[predict_label == pred_i] = target_i
    reordered_preds[predict_label == rid[1]] = 1 # coarsively label the object as the sea clutter for evalutaion acc and miou.

    acc = _acc(reordered_preds, gt_label,  gt_k)  # omit the object's echos. acc and miou only for 0 and 1 labels.

    miou =_miou(reordered_preds, gt_label, gt_k) # note that only consider the miou of sea clutter and background
    return acc, miou

def reorder_segmentation(predict_label, gt_label):
    org_shape = gt_label.shape
    assert (predict_label.numel() == gt_label.numel())
    predict_label = predict_label.reshape(predict_label.numel())  # vectorize the prediction and gt label
    gt_label = gt_label.view(gt_label.numel())

    num_samples = predict_label.shape[0]

    match = _hungarian_match(predict_label, gt_label, num_k=2)

    reordered_preds = torch.zeros_like(predict_label)
    for pred_i, target_i in match:
        reordered_preds[predict_label == pred_i] = target_i

    re_assign = reordered_preds.reshape(org_shape)
    return re_assign

def evaluate_segmentation(predict_label, gt_label, gt_k=3, verbose=0):
    '''
    First need to match predict_label's category with the gt_label.
    e,g. the land in prediction may be marked with 1, while in gt_label it is marked with 2.
    Through hungry_match, find the maximum label_match pair between prediction and gt.
    :param precit_label:
    :param gt_label:
    :return:
    '''
    assert(predict_label.numel() == gt_label.numel())
    predict_label = predict_label.reshape(predict_label.numel()) #vectorize the prediction and gt label
    gt_label      = gt_label.view(gt_label.numel())

    num_samples      = predict_label.shape[0]

    if verbose == 2:
        print("num_test: %d" % num_samples)
        for c in range(gt_k):
            print("gt_k: %d count: %d" % (c, (gt_label == c).sum()))

    match = _hungarian_match(predict_label, gt_label, gt_k)

    reordered_preds = torch.zeros_like(predict_label)

    for pred_i, target_i in match:
        reordered_preds[predict_label == pred_i] = target_i
        if verbose == 2:
            print((pred_i, target_i))
    acc = _acc(reordered_preds, gt_label, gt_k)
    miou =_miou(reordered_preds, gt_label, gt_k)
    return acc, miou


def re_assign_label(predict_label, gt_label, gt_k=2):
    '''
    First need to match predict_label's category with the gt_label.
    e,g. the land in prediction may be marked with 1, while in gt_label it is marked with 2.
    Through hungry_match, find the maximum label_match pair between prediction and gt.
    :param predict_label:
    :param gt_lablel:
    :param gt_k:
    :return:
    '''

    # ## soft re-assign the label based on hungarian match [get best miou or acc]
    # org_shape = gt_label.shape
    # assert (predict_label.numel() == gt_label.numel())
    # predict_label = predict_label.view(predict_label.numel())  # vectorize the prediction and gt label
    # gt_label = gt_label.view(gt_label.numel())
    # reordered_preds = torch.zeros_like(predict_label)
    # # match = _hungarian_match(predict_label, gt_label, gt_k)
    # # for pred_i, target_i in match:
    # #     reordered_preds[predict_label == pred_i] = target_i

    ## hard re-assign the label based on target coverage [get best detection rate]
    reordered_preds = 1- predict_label
    # bg_mask = (predict_label == 0)
    # fg_mask = (predict_label == 1)
    # if torch.sum(bg_mask * (gt_label==1)) > torch.sum(fg_mask * (gt_label==1)): #reverse label.
    #     reordered_preds[bg_mask] = 1
    #     reordered_preds[fg_mask] = 0

    pred_acc = _acc(predict_label, gt_label, gt_k)
    reor_acc = _acc(reordered_preds, gt_label, gt_k)

    pred_far = _false_alarm_rate(predict_label, gt_label)
    reor_far = _false_alarm_rate(reordered_preds, gt_label)

    pred_target_iou = _target_iou(predict_label, gt_label)
    reor_target_iou = _target_iou(reordered_preds, gt_label)

    if (pred_acc < reor_acc):
    #if (pred_far > reor_far): #using false alarm rate to re-assign the label.
    # if (pred_target_iou < reor_target_iou): #using target iou to re-assign the label.
         return reordered_preds
    else:
        return predict_label
    #assert(pred_dr <= relabel_dr)   # the re-assign label should not decrease the detection rate.
    #return reordered_preds

def get_psnr(img, label):
    '''
    :param img:  the source image
    :param label: the label on true target
    :return:
    '''
    #img = img.squeeze(dim=1)
    assert(img.shape == label.shape)
    target = img*label
    target_pixels = torch.sum(label)
    peak   = torch.max(target)
    target_power = torch.sum(target ** 2) / target_pixels

    back  = img - target
    # Erc: average clutter energy.
    erc = torch.sum(back ** 2) / (img.numel()-target_pixels)

    psnr = 10 * torch.log10(peak ** 2 / erc)
    snr  = 10 * torch.log10(target_power / erc)
    return psnr.item(), snr.item()


def  show_segmentation(X, pred_label, label, str, config):
     '''
     show segmentation results by plt.imshow. for at most 5 images
     :param X:
     :param pred_label:
     :param label:
     :return:
     '''
     nums = X.shape[0]
     nums = min(nums, 5)
     if config.dataset == 'Potsdam':
         fig_org, axes = plt.subplots(3, 5, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(6, 8))
         fig_org.suptitle('src_gt_pred_%s' % str)
         for i in range(nums):
             img = X[i,::].permute([1,2,0]).cpu().numpy()
             axes[0,i].imshow(img[:,:,:3]) # X [B, C, H, W] -> [H, W, C], show only rgb channels.
             axes[1,i].imshow(label[i,::].cpu().numpy())
             axes[2,i].imshow(pred_label[i,::].cpu().numpy())
             for j in range(3):
                 axes[j,i].set_xticklabels([])
                 axes[j,i].set_yticklabels([])
         fig_org.savefig(os.path.join(config.out_root, 'src_gt_out_%s.png' % str), bbox_inches='tight')
     if config.dataset == 'nau_radar' and X.shape[1]<3:
        fig_org, axes = plt.subplots(4, 4, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8),sharex=True, sharey=True)
        fig_org.suptitle('src_chn_gt_pred_%s' % str)
        nums = min(nums, 4)
        for i in range(nums):
            img = X[i, ::].cpu().numpy()
            axes[0, i].imshow(img[0, :, :])  # 0 channel contains echos
            if X.shape[1]> 1:
                axes[1, i].imshow(img[1, :, :])  # 1 channel contains adtional information
            axes[2, i].imshow(label[i, ::].cpu().numpy())
            axes[3, i].imshow(pred_label[i, ::].cpu().numpy())
            for j in range(3):
                axes[j, i].set_xticklabels([])
                axes[j, i].set_yticklabels([])
        fig_org.savefig(os.path.join(config.out_root, 'src_chn_gt_out_%s.png' % str), bbox_inches='tight')
     if config.dataset == 'nau_radar' and X.shape[1]==3: # using global fno information
        fig_org, axes = plt.subplots(5, 5, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8),sharex=True, sharey=True)
        fig_org.suptitle('src_chn_gt_pred_%s' % str)
        nums = min(nums, 5)
        for i in range(nums):
            img = X[i, ::].cpu().numpy()
            axes[0, i].imshow(img[0, :, :])  # 0 channel contains echos
            axes[1, i].imshow(img[1, :, :])  # 1 channel contains H0 of fno
            axes[2, i].imshow(img[1, :, :])  # 2 channel contains H1 of fno
            axes[3, i].imshow(label[i, ::].cpu().numpy())
            axes[4, i].imshow(pred_label[i, ::].cpu().numpy())
            for j in range(3):
                axes[j, i].set_xticklabels([])
                axes[j, i].set_yticklabels([])
        fig_org.savefig(os.path.join(config.out_root, 'src_h0_h1_gt_out_%s.png' % str), bbox_inches='tight')
     plt.show()
     plt.close(fig_org)

def show_unet_adversarial(X, pred_t, pred_d, label, str, config):
    fig_org, axes = plt.subplots(4, 4, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), sharex=True, sharey=True)
    fig_org.suptitle('src_gt_predTop_predDown_%s' % str)
    nums = X.shape[0]
    nums = min(nums, 4)
    for i in range(nums):
        img = X[i, ::].cpu().numpy()
        axes[0, i].imshow(img[0, :, :])  # 0 channel contains echos
        axes[1, i].imshow(label[i, ::].cpu().numpy())
        axes[2, i].imshow(pred_t[i, ::].cpu().numpy())
        axes[3, i].imshow(pred_d[i, ::].cpu().numpy())
        for j in range(4):
            axes[j, i].set_xticklabels([])
            axes[j, i].set_yticklabels([])
    plt.show()
    fig_org.savefig(os.path.join(config.out_root, 'src_gt_predTop_predDown_%s.png' % str), bbox_inches='tight')
    plt.close(fig_org)

def show_onet_img(tensor_list, str, config):
    X = tensor_list[0]
    bsz  = X.shape[0]
    tlen = len(tensor_list)
    nums = min(bsz, tlen)
    fig_org, axes = plt.subplots(nums, nums, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), sharex=True,
                                 sharey=True)
    fig_org.suptitle('%s' % str)
    for ax_row in range(nums):
        for ax_col in range(nums):
            ele = tensor_list[ax_row]
            if ele.dim()==4:
                assert ele.shape[1]==1
                axes[ax_row, ax_col].imshow(ele.squeeze(dim=1).cpu()[ax_col,::])
            if ele.dim()==3:
                axes[ax_row, ax_col].imshow(ele.cpu()[ax_col, ::])
            axes[ax_row, ax_col].set_xticklabels([])
            axes[ax_row, ax_col].set_yticklabels([])
    #plt.show()
    fig_org.savefig(os.path.join(config.out_root, '%s.png' % str), bbox_inches='tight')
    plt.close(fig_org)

def show_unet_adversarial_v2(X, pred_t, pred_d, label, pred_label, str, config):
    fig_org, axes = plt.subplots(5, 5, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), sharex=True, sharey=True)
    #fig_org, axes = plt.subplots(5, 8, figsize=(5, 8), sharex=True, sharey=True)
    fig_org.suptitle('%s' % str)
    nums = X.shape[0]
    nums = min(nums, 5)
    for i in range(nums):
        img = X[i, ::].cpu().numpy()
        axes[0, i].imshow(img[0, :, :])  # 0 channel contains echos
        axes[1, i].imshow(label[i, ::].cpu().numpy()) # gt
        axes[2, i].imshow(pred_label[i, ::].cpu().numpy()) # predict_label
        axes[3, i].imshow(pred_t[i, ::].cpu().numpy()) # predict_top
        axes[4, i].imshow(pred_d[i, ::].cpu().numpy()) # predict_down
        for j in range(5):
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
    #plt.show()
    fig_org.savefig(os.path.join(config.out_root, '%s.png' % str), bbox_inches='tight')
    print('save resulst to %s.png' % os.path.join(config.out_root, '%s.png' % str))
    plt.close(fig_org)

def show_nau_rain(X, names, pred_t, pred_d, label, pred_label, str, config):
    fig_org, axes = plt.subplots(5, 5, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8), sharex=True, sharey=True)
    #fig_org, axes = plt.subplots(5, 8, figsize=(5, 8), sharex=True, sharey=True)
    fig_org.suptitle('%s' % str)
    nums = X.shape[0]
    nums = min(nums, 5)
    for i in range(nums):
        img = X[i, ::].cpu().numpy()
        axes[0, i].imshow(img[0, :, :])  # 0 channel contains echos
        axes[1, i].imshow(label[i, ::].cpu().numpy()) # gt
        axes[2, i].imshow(pred_label[i, ::].cpu().numpy()) # predict_label
        axes[3, i].imshow(pred_t[i, ::].cpu().numpy()) # predict_top
        axes[4, i].imshow(pred_d[i, ::].cpu().numpy()) # predict_down
        for j in range(5):
            time, fid, nick = names[i].split('_')
            if j == 0:  # the first row show the image id, acc, miou
                sub_title = '%s\n%s_%s' % (time, fid, nick)
                axes[j, i].set_title(sub_title, fontsize=8)
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
    plt.show()
    fig_org.savefig(os.path.join(config.out_root, '%s.png' % str), bbox_inches='tight')
    plt.close(fig_org)


def show_unet_2ndstage_test(X1, X2, fg, label, label1, label2, txt, config):
    fig_org, axes = plt.subplots(6, 5, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8*5/6, 8), sharex=True, sharey=True)
    #fig_org, axes = plt.subplots(5, 8, figsize=(5, 8), sharex=True, sharey=True)
    fig_org.suptitle('%s_x1_x2_fg_label_gt12' % txt)
    nums = X1.shape[0]
    nums = min(nums, 5)
    for i in range(nums):
        axes[0, i].imshow(X1[i,0, :, :].cpu().numpy())  # input image
        axes[1, i].imshow(X2[i,0, :, :].cpu().numpy()) #  fg of 1st stage output
        axes[2, i].imshow(fg[i,0, :, :].cpu().numpy()) # fg
        axes[3, i].imshow(label[i, ::].cpu().numpy())  # gt_label
        axes[4, i].imshow(label1[i, ::].cpu().numpy()) # predict_label in 1st stage
        axes[5, i].imshow(label2[i, ::].cpu().numpy())  # predict_label in 2nd stage
        for j in range(6):
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
    plt.show()
    fig_org.savefig(os.path.join(config.out_root, '%s.png' % txt), bbox_inches='tight')
    plt.close(fig_org)

def show_nau_train_result(res_list_dict,config):

    loss_list = res_list_dict['loss']
    acc_list  = res_list_dict['pixel_acc']
    miou_list = res_list_dict['miou']

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(loss_list, 'b', label='train_loss')
    axs[0].set_title('train_loss vs epochs')
    #axs[0,0].xlabel('epochs')

    axs[1].plot(acc_list, 'r-', label='pixel_acc')
    axs[1].plot(miou_list, 'g', label='miou_list')
    axs[1].set_xlabel('epochs')
    plt.legend()
    #sys.stdout = old_stdout #redirect the stdout
    #log_file.close()
    plt.show()
    epochs = len(loss_list)
    fig.savefig(os.path.join(config.out_root, 'training_%d_epochs_results.png'%epochs), bbox_inches='tight')
def array_normal(input):
    '''
    resacle input's value in the range of [0,1]
    :param input:
    :return:
    '''
    if input.max()==input.min(): # all white or all black image
        input = input.max()/(input.max()+np.spacing(1)) # equal to 1. or 0.
    else:
        input = (input - input.min())/(input.max() - input.min() + np.spacing(1))
        #input = (input - 0) / (input.max() - 0 + np.spacing(1))
    #input = input.astype(np.float32)
    return input
def tensor_normal_per_frame(input:torch.Tensor):
    #Scale tensor to [0,1] in H,W dimension of a frame.
    #Make gradient-computing available, avoid in-place computing. e.g. input[i,j,::] = array_normal(input[i,j,::])
    assert (input.dim() == 4)
    nb, nc, h, w = input.shape[0:4]
    # input_eq = input.clone()
    # for i in range(nb):
    #     for j in range(nc):
    #         input_eq[i,j,::] = array_normal(input_eq[i,j,::])
    # return input
    input_vec = input.reshape(nb, nc, h*w)
    input_min = input_vec.min(dim=-1, keepdim=True)[0] #element 0 takes min values in original shape
    input_max = input_vec.max(dim=-1, keepdim=True)[0]

    output = (input_vec - input_min)/(input_max - input_min + np.spacing(1))
    #output = output.reshape(nb,nc, h, w)
    return output.reshape(nb,nc, h, w)

from collections import namedtuple
def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            if isinstance(output, tuple):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details
#nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free