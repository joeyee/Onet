'''
Generate Rayleigh-distributed sea clutter background and Extended Objects in Gaussian diffusion.
Based on my work of TAES_code2021's file 'motion_simulation_k_distribution_20210923.py'

K-distributed simulation is also added in this file with the "kdist" option.
Created by ZhouYi@Provence_Dalian on 20230208.
'''

import os
from   datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
import torchvision.transforms as transforms
import K_distributed_SeaClutter_Simulation_20210919 as kmodel
import torch
import utils_20231218 as uti

from scipy.stats    import rayleigh,chi,chi2

local_snrs  = []
global_snrs = []

dec_str = [12  ,  11  , 10  , 9   , 8   , 7   , 6   , 5   , 4   , 3   , 2   , 1   , 0   , -1  , -2  ]

def gaussian_kernel2d(sigma_x, sigma_y, theta, bnorm=True):
    '''
    Return a 2d Gaussian kernel template (2d matrix) with orientation.
    :param sigma_x:
    :param sigma_y:
    :param theta: rotation theta of 2d Gaussian
    :return: Gaussian Kernel Template.
    '''
    kernel_wr = np.int32(sigma_x * 2.5 + 0.5)
    kernel_hr = np.int32(sigma_y * 2.5 + 0.5)

    #if kernel_hr < 5 or kernel_wr < 5:

    #    raise ValueError('kenrel width or/and height are too small')

    kx = np.arange(-kernel_wr, kernel_wr + 1)
    ky = np.arange(-kernel_hr, kernel_hr + 1)
    KX, KY = np.meshgrid(kx, ky)
    theta = -1*theta

    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta)  / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)
    # f(x,y)=Aexp(−(a(x−xo)2+2b(x−xo)(y−yo)+c(y−yo)2)) , here xo=0, yo=0
    # f(x,y)=Aexp(−(ax^2+2bxy+cy^2))
    # a   = cos2θ2σ2X + sin2θ2σ2Y
    # b   =−sin2θ4σ2X + sin2θ4σ2Y
    # c   = sin2θ2σ2X + cos2θ2σ2Y

    kgauss = np.exp(-(a * KX ** 2 + 2 * b * KX * KY + c * KY ** 2))
    if bnorm:#normalization in default mode.
        kgauss = kgauss / np.sum(kgauss)
    return kgauss


def add_gaussian_template_on_clutter_v3(cx, cy, w, h, theta, erc, snr, clutter_background, foreground_mask, swerling_type=0):
    '''
    Rewrite the swerling type's pdf. kgauss is normalized.
    :return:
    '''
    # Erc: average clutter energy.
    # Erc = np.sum(clutter_background ** 2) / clutter_background.size
    sigma_x = (w/2  - 0.5) / 2  # sigma_x is related to the width of the template
    sigma_y = (h/2  - 0.5) / 2

    kgauss    = gaussian_kernel2d(sigma_x, sigma_y, theta, bnorm=False)  # Get diffusive coefficients for a 2d gaussian
    Egk_numer = np.sum(kgauss.ravel() ** 2) / kgauss.size  # 2d gaussian's average power.

    h_t, w_t  = kgauss.shape
    ly = int(cy - (h_t - 1) / 2)
    ry = int(cy + (h_t - 1) / 2)
    lx = int(cx - (w_t - 1) / 2)
    rx = int(cx + (w_t - 1) / 2)

    img_h, img_w = clutter_background.shape
    if ly < 0 or lx < 0 or ry > img_h or rx > img_w:
        raise ValueError('template location is beyond the image boundaries!')
    bk_roi = clutter_background[ly:ly + h_t, lx:lx + w_t]
    # compute the amplitude coefficients according to the SNR Eq.
    kcoef_global = np.sqrt(np.power(10, (snr / 10)) * erc / Egk_numer)

    kcoef_peak = np.sqrt(np.power(10, (snr / 10)) * erc) # point's snr reversion
    # average power of clutter is computed by numerical results in local roi-window.
    erc_local = np.sum(bk_roi ** 2) / bk_roi.size
    kcoef_local = np.sqrt(np.power(10, (snr / 10)) * erc_local / Egk_numer)

    kcoef = kcoef_peak
    if swerling_type == 0:  # swerling type 0 target
        kcoef_t  = kcoef
        template = kgauss * kcoef_t
    if swerling_type == 1:
        ray_scale = kcoef/np.sqrt(2)#choosing mode  # /np.sqrt(2)
        # central amplitude obeys the rayleigh distribution, which 2*sigma^2 = sigma_t = kcoef**2 (swerling_0's Amplitude)
        kcoefs = rayleigh.rvs(loc=0, scale=ray_scale, size=1000)
        kcoef_t = np.mean(kcoefs)
        template = kgauss * kcoef_t
    if swerling_type == 3:  # central amplitude obeys the chi distribution, which degrees of freedom k=4.
        df = 4
        chi2_scale= kcoef/np.sqrt(df*2+df**2)#np.sqrt(df-2)#
        kcoefs    = chi2.rvs(df=df, scale=chi2_scale,  size=1000)# or kcoef_t  = chi2.rvs(df=kcoef, size=1), then template=kgauss*kcoef
        kcoef_t   = np.mean(kcoefs)
        template  = kgauss * (kcoef_t) #

    # Get decrease_coeffient to make sure the inner gaussian template satisfy the snr requirement.
    tcx, tcy = w_t / 2, h_t / 2
    snr_lis = list(range(12, -3, -1))  # [12, 11, ..., -1, -2]
    # shrink rate, take from cfar results.
    snr_lis = [12,     11,   10,    9,    8,    7,    6,    5,    4,    3,    2,    1,    0,  -1,  -2]
    wr_lis  = [1.62, 1.67, 1.65, 1.76, 1.80, 2.00, 2.20, 2.30, 3.20, 3.50, 3.70, 3.90, 4.00, 4.2,  4.5]
    hr_lis  = [0.88, 0.89, 0.90, 0.92, 1.00, 1.10, 1.20, 1.20, 1.55, 1.55, 1.65, 1.70, 1.75, 2.0,  2.5]
    incs_sw1= np.linspace(1.00, 2.55, 15)#[0.95, 1.00, 0.90, 0.85, 0.80, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 2.00, 2.00, 2.20, 2.50]
    #incs_sw1 = np.log2(1+incs_sw1)
    decs    = np.linspace(0.78, 0.34, 15)
    #decs_sw1= np.linspace(1.00, 0.45, 15)
    decs_sw3= np.linspace(1.20, 0.30, 15)
    # decrease the size of Gaussian template, similar to the cfar_seg results.
    # [cfar shrink the real target, when outside is lower than center]
    wr = wr_lis[snr_lis.index(snr)]
    hr = hr_lis[snr_lis.index(snr)]
    iw, ih = w_t / wr, min(h_t / hr, h_t)
    ix, iy, iw, ih = np.intp([tcx - iw / 2, tcy - ih / 2, iw, ih])
    inner_gauss = template[iy:iy + ih, ix:ix + iw]

    #dec_coef = np.sqrt(np.power(10, (snr / 10)) * erc_local / np.mean(inner_gauss ** 2))
    #dec_str[snr_lis.index(snr)] = '%.2f' % dec_coef

    if swerling_type == 0: # decreasing for non-fluctuating target type
        dec_coef = decs[snr_lis.index(snr)]
        template = template * 1#dec_coef  # np.sqrt(1.618) #/2.8 # Make sure that in shrinked (cfar-segmented) target region still holds low snr.
    if swerling_type == 1:
        inc_coef = incs_sw1[snr_lis.index(snr)]
        template = template * 1   #inc_coef
    if swerling_type == 3:
        dec_coef = decs_sw3[snr_lis.index(snr)]
        template = template * 1#dec_coef
    loc_snr  = 10 * np.log10(np.sum(template ** 2) / np.sum(bk_roi ** 2))
    glob_snr = 10 * np.log10(np.sum(template ** 2) / (erc * template.size))
    peak_snr = 10 * np.log10(np.max(template)**2   /  erc) #point's snr

    # print('Swerling Type %d, kcoef_t %.2f (w %d, h %d), extened_egk %.2E' % (swerling_type, kcoef_t, w, h, Egk_numer))
    # print('average (target - local clutter) power is (%.2f - %.2f)' % (np.sum(template ** 2) / template.size, erc_local))
    # print('Asked snr is %d, simulated local snr is %.2f, simulated global snr is %.2f' % (snr, loc_snr, glob_snr))
    local_snrs.append(loc_snr)
    global_snrs.append(peak_snr)

    #----actual snr is about set_snr*2
    #template_mask = kgauss > (kgauss.max()-3*kgauss.std())
    template_mask = kgauss > (kgauss.max() - 2 * kgauss.std())
    foreground = (template > bk_roi) * template   # Onet's version after 2023_05_11
    # #foreground    = template_mask * template     # Onet's version before 2023_05_11
    clutter_background[ly:ly + h_t, lx:lx + w_t] = foreground + bk_roi

    #----actual snr is about set_snr/2
    # template_mask = template > bk_roi
    # foreground = (template > bk_roi) * template+ bk_roi * (template <= bk_roi)
    # clutter_background[ly:ly + h_t, lx:lx + w_t] = foreground
    #-------------------

    foreground_mask[ly:ly + h_t, lx:lx + w_t]   += template_mask
    foreground_mask = foreground_mask>0
    #clutter_background[ly:ly + h_t, lx:lx + w_t] = template + bk_roi

    #Real_SNR is normally higher than peak_snr
    real_snr = 10 * np.log10(max(np.max(template + bk_roi)-np.sqrt(2), np.spacing(1)) / 2)
    # cmin = clutter_background.min()
    # cmax = clutter_background.max()
    # clutter_background = (clutter_background-cmin)/(cmax-cmin+np.spacing(1))
    return clutter_background, foreground_mask

def get_k_frame(snr=10):
    '''
    Get one frame combine targets and k-distributed sea clutter together.
    #add swerling type on Mar 2, 2021.
    #add correlated k-distributed sea clutter on Sep 23, 2021.
    :param frame_no:
    :return:
    '''
    # make bk in bigger images 400x400. Then do center crop transform to 200x200 for Onet networks.
    img_sz = (400, 400)
    img_h, img_w = img_sz

    k_background, _ = kmodel.generate_K_distributed_noise(img_h, img_w, gamma_shape=5)
    k_background    = k_background.astype(np.float32)

    erc      = np.sum(k_background ** 2) / k_background.size
    # add targets on the simulated position in each frame
    bg_frame = k_background
    fg_mask  = np.zeros_like(bg_frame)

    # Each frame gets multiple targets.
    target_num = 20  # target numbers.
    img_cx, img_cy = (img_sz[0] / 2, img_sz[1] / 2)
    cx = np.random.normal(img_cx, 30, target_num)
    cy = np.random.normal(img_cy, 24, target_num)
    w = np.random.normal(10, 2, target_num)
    h = np.random.normal(18, 2, target_num)
    theta = np.random.rand(target_num) * 180
    swerling_type = 0

    for i in range(target_num):
        bg_frame, fg_mask = add_gaussian_template_on_clutter_v3(cx[i], cy[i], w[i], h[i], theta[i], erc, snr,
                                                                bg_frame, fg_mask, swerling_type)
    # plt.imshow(bg_frame)
    # plt.figure()
    # plt.imshow(fg_mask)
    # plt.show()
    # exit(0)
    # print()
    return bg_frame, fg_mask


def get_rayleigh_frame(snr=10):
    #make bk in bigger images 400x400. Then do center crop transform to 200x200 for Onet networks, 224x224 for onet2.0.
    img_sz = (400, 400)
    ray_background = rayleigh.rvs(loc=0, scale=1, size=img_sz) #sigma_n=E(n^2) = 2*scale^2
    # Erc: average clutter energy.
    erc = np.sum(ray_background ** 2) / ray_background.size
    #add targets on the simulated position in each frame
    bg_frame = ray_background
    fg_mask  = np.zeros_like(bg_frame)

    # Each frame gets multiple targets.
    targets = {}
    target_num = 20 # target numbers.
    img_cx, img_cy=(img_sz[0]/2, img_sz[1]/2)
    cx = np.random.normal(img_cx, 30, target_num)
    cy = np.random.normal(img_cy, 24, target_num)
    w  = np.random.normal(10,  2,  target_num)
    h  = np.random.normal(18,  2,  target_num)
    theta=np.random.rand(target_num)*180
    swerling_type=0

    for i in range(target_num):
        bg_frame, fg_mask = add_gaussian_template_on_clutter_v3(cx[i], cy[i], w[i], h[i], theta[i], erc, snr,
                                                               bg_frame, fg_mask, swerling_type)
        #print('w, h, area', int(w[i]), int(h[i]), int(w[i]*h[i]))
    # plt.imshow(bg_frame)
    # plt.figure()
    # plt.imshow(fg_mask)
    # plt.show()
    # print()
    return bg_frame, fg_mask

def prepare_frames(type= 'rayleigh', fnums=4, snr=10):
    '''
    transfer multiple frames to tensor format
    :return:
    '''
    for i in range(fnums):
        if type=='rayleigh':
            img, label = get_rayleigh_frame(snr)
        if type=='kdist': #k distributed
            img, label = get_k_frame(snr)
        img   = torch.tensor(img).unsqueeze(dim=0)
        img   = uti.array_normal(img) # normalize to [0,1] range
        label = torch.tensor(label).unsqueeze(dim=0)
        if i==0:
            imgs = img
            labels= label
        else:
            imgs = torch.concat([imgs, img], dim=0)
            labels = torch.concat([labels, label], dim=0)
    imgs   = imgs.unsqueeze(dim=1).to(torch.float32)
    labels = labels.to(torch.float32)
    scr    = get_scr(imgs, labels)
    print('Wanted %s clutter SNR: %d, Simulated Average (extended region -- peak point) SNR is (%.2f - %.2f), SCR %.2f at %s'
          % (type, snr, np.mean(local_snrs), np.mean(global_snrs), scr, datetime.now()))
    return imgs, labels

def get_scr(image, label):
    '''
    compute the scr of the extended target in the clutter.
    :param image:
    :param label:
    :return:
    '''
    #if image is tensor
    if isinstance(image, torch.Tensor):
        signal_mean_energy = torch.sum((label*image) ** 2) / torch.sum(label==1)
        noise_mean_energy  = torch.sum(((1-label)*image) ** 2) / torch.sum(label==0)
        scr = 10*torch.log10(signal_mean_energy/noise_mean_energy)
        scr = scr.item()
    else:
        signal_mean_energy = np.sum((label*image) ** 2) / np.sum(label==1)
        noise_mean_energy  = np.sum(((1-label)*image) ** 2) / np.sum(label==0)
        scr = 10*np.log10(signal_mean_energy/noise_mean_energy)
    return scr
def prepare_data(img_sz=(224,224), bg_type='rayleigh', file_name=None):
    '''
    bg_type in [rayleigh, kdist]
    :param img_sz:
    :param bg_type: 'rayleigh' distributed and 'kdist' distributed sea clutter with extended targets in 1050 frames in 10 snr kinds.
    :return:
    '''
    transform = transforms.CenterCrop(img_sz)
    psnrs = []
    #for snr in [0, 2, 4, 5, 6, 8, 10]: # get 7 different snrs.
    for psnr in range(0,11): # Get 11 different snrs.
        print('frame psnr is ', psnr)
        imgs, labels = prepare_frames(type=bg_type, fnums = 150, snr=psnr)
        imgs =  transform (imgs)
        labels = transform(labels)
        if psnr==0:
            distributed_imgs  = imgs
            distributed__labels= labels
        else:
            distributed_imgs = torch.concat((distributed_imgs, imgs), dim=0)
            distributed__labels=torch.concat((distributed__labels, labels), dim=0)
        psnrs.extend([psnr]*150)
    data_dict = {'%s_imgs'%bg_type:distributed_imgs, '%s_labels'%bg_type:distributed__labels, 'psnr':psnrs,
                     'desc':'%s clutter add 20 extended targets [pure fg higher than mu-2*simga] in each frame '
                            'with snr0-10. Each snr get 150 frames [total 150x11=1650 frames] .'%bg_type}
    #file_name = '/home/ubuntu/datasets/sim_background/%s_pure_fg_data_2sigma.pt'%bg_type          # data before 20230512
    #file_name = '/home/ubuntu/datasets/sim_background/%s_bg_0-10_PSNR_2dGaussian_fg.pt' % bg_type # data after 20230512
    #file_name ='/root/datasets/sim_background/%s_pure_fg_data_2sigma.pt'%bg_type                   # datap for Onet2.0 on 20240411
    #add file_name to save data for published Onet on 20250407
    torch.save(data_dict, file_name) #1050 frames = 321M
    print('%s is saved' % file_name)
if __name__=='__main__':
    #nohup python -u Rayleigh_bg_Gaussian_EOT_generator_20230208.py  > ./checkpoint/sim_clutter/generate_frame.log &
    import configs.config_tip2022_20230411    as conf_model  # load configuration from '.yml' file.
    print('current pid: ', os.getpid())
    simbg_config   = conf_model.generate_config('./configs/train_onet_20250407.yml', dataset_name='Rayleigh')
    if not os.path.exists(simbg_config.dataset_root):
        os.makedirs(simbg_config.dataset_root)
    data_file_name = os.path.join(simbg_config.dataset_root, simbg_config.data_file_name)
    #get_rayleigh_frame()
    prepare_data(bg_type='rayleigh', img_sz=(224,224), file_name=data_file_name)
    #prepare_data(bg_type='kdist', img_sz=(200, 200))
    print('Done')
    exit(0)
