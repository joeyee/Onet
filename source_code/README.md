# Onet Project User Manual

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Dependencies and Setup](#dependencies-and-setup)
4. [Training Guide](#training-guide)

## Project Overview
This project implements Onet (unsupervised semantic segmentation) for various scenarios including:
- Simulated Rayleigh distributed clutter
- NAU-Rain case
- ZY3 remote sensing datasets

## File Structure
```
onet_github/
├── configs/                 # Configuration files
│   ├── train_onet_20250407.yml    # Main training configuration
│   └── config_tip2022_20230411.py # Configuration utilities
├── checkpoint/             # Saved model checkpoints
│   ├── sim_clutter/       # Simulated clutter models
│   └── nau_rain/          # NAU-Rain models
│   └── zy3/          # zy3 models
├── dataloader/            # Data loading utilities
│   ├── nau_rain_20230523.py      # NAU-Rain dataset loader
│   ├── simbg4onet_20230209.py    # Simulated background loader
│   └── zy3_cloud_thumbnailv5_20240304.py  # ZY3 dataset loader
├── Onet_vanilla_20240606.py    # Main ONET model implementation
├── Train_Onet_on_simclutter_20250407.py    # Training for simulated clutter
├── Train_Onet_on_zy3_20240606.py    # Training for ZY3 datasets
├── exp_nau_rain_20240513.py    # NAU-Rain experiment
├── utils_20231218.py      # Utility functions
├── haze_remove_20240313.py      # hazing remove
├── test_pre_processing_on_zy3_testset_20240607.py  # test pre-processing method for zy3 
└── README.md              # Project documentation
```

## Dependencies and Setup

### Required Datasets and Paths
Download link for the [datasets](https://pan.baidu.com/s/1G8Wq4HCYaYVXuYnNE0uT7w?pwd=dlmu)
1. Simulated Rayleigh Clutter Dataset:
   - Path: `/datasets/sim_background/`
   - Main data file: `rayleigh_2sigma.pt`
   - Contains: Generated clutter data with 11 SNR levels from 0 to 10.
   - Each SNR level has 150 images (total 1650 images)
   - Generated using: `Rayleigh_bg_Gaussian_EOT_generator_20230208.py`

2. ZY3 Dataset:
   - Path: `/datasets/zy3-thumbnails224/`
   - Contains: Remote sensing images
   - Image size: 224x224x3
   - Data Files: 'zy3_thumbnail224_test_label_dict50_v2.pt', 'zy3_thumbnail224_train_dict250_v2.pt' for 1st round training onet.
   - load the source files of zy3 in '/test-imgs' and '/test_label_50_255' and using 'test_pre_processing_on_zy3_testset_20240607.py' to generate 'zy3_thumbnail224_test_label_dict50_bestACC_preprocess.pt' which is more contrastive for onet inferencing.
   - The original dataset (zy3-datasets-source) is used in the following papers:
   [1]J. Guo, J. Yang, H. Yue, X. Liu and K. Li, "Unsupervised Domain-Invariant Feature Learning for Cloud Detection of Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022
   [2]J. Guo, J. Yang, H. Yue and K. Li, "Unsupervised Domain Adaptation for Cloud Detection Based on Grouped Features Alignment and Entropy Minimization," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022
   - For using zy3 in Onet, we shrink the original zy3 into 224x224x3 thumbnails, some image pre-processing (histogram equalization, contrast enhancement) are conducted for better contrast. Onet can are trained in the small thumbnail data files in a few epochs.

3. NAU-Rain Dataset:
   - Path: `/datasets/nau_rain/`
   - Main data file: `naurain_20200819144753_10_target_img_label_dict.pt`
   - Contains: Rain-affected images with labels
   - Image size: 200x200

### Output Directories
- Model files: '/onet_project_dir/checkpoint/' for sim_clutter, zy3 and nau_rain


## Training Guide

### 1. Training ONET for Simulated Rayleigh distributed Clutter
```bash
python Train_Onet_on_simclutter_20250407.py
```
This script trains ONET on simulated Rayleigh distributed clutter with the following features:
- Uses `rayleigh_2sigma.pt` dataset
- using only low snr (from 0 to 2) frames 
- Outputs saved in `/onet_project_dir/checkpoint/sim_clutter/`

### 2. Training ONET for ZY3 Datasets
```bash
python Train_Onet_on_zy3_20240606.py
```
This script trains ONET on ZY3 remote sensing images with:
- 11 epochs training
- 224x224x3 image shape
- Output saved in '/onet_project_dir/checkpoint/zy3'

Please note that Onet can be trained in only 11 epoches (less than 5 minutes) for zy3 with OA(0.8833) and miou(0.7218). Then we can using pre-processing techniques to improve the contrast and enhance the thin clouds for the remote sensing images (please check test_pre_processing_on_zy3_testset_20240607.py). After that using the 11-epoch trained onet model, the OA reaches 0.9254 and miou raises to 0.7958. We believe the Onet structure enables the fast training ability and it has the potential to be an efficient label tool for marking the regions with strong intensities. 



### 3. Training ONET for NAU-Rain Case
```bash
python exp_nau_rain_20240513.py
```
```
This evaluates the model's performance on NAU-Rain dataset.
- model weights extracted from onet_lowsnr0-2 which is trained in the Simulated Rayleigh distributed Clutter with low snr (from 0 to 2).
- 200x200 image size
- target segmentation in rain clutter
- Uses `naurain_20200819144753_10_target_img_label_dict.pt` dataset
- Output saved in '/onet_project_dir/checkpoint/nau_rain'


## Tips and Best Practices
1. Always check the configuration files in `configs/` before starting training
2. Monitor training progress using the logs in '*.log' files in the output directory of each task.
3. Use the pre-processing steps for better model performance in zy3.
4. Save model checkpoints regularly during training
