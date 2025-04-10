'''
United configure file for Onet, Infoseg, IIC and supervised Unet.
Get pre-setting parameters from .yml files and convert it to argparse.

Created by ZhouYi@Provence_Dalian on 2023/04/11
'''

import yaml
import argparse
import os
import torch
import utils_20231218 as uti
import pickle

def setup_config(conf_yml, dataset='zy3'):
    """Sets up the configs
    Returns:
        ArgumentParser -- parser with arguments set.
    """
    with open(conf_yml, 'r') as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)
        config_dataset = config_dict[dataset]

        parser = argparse.ArgumentParser()
        parser.add_argument('--enc_in_channels', type=int, help='depth of channels in the beginning layer of encoding')
        parser.add_argument('--outc_channels',   type=int, help='depth of channels at the end of the decoding layer')
        parser.add_argument('--layer_type', type=str, help='type of layer')
        parser.add_argument('--feature_src', type=str, help='source of feature from encoding or decoding end') # 'enc' or 'dec'
        parser.add_argument('--enc_depth', type=int, help='depth of encoding layers')
        #parser.add_argument('--share_weight',  default=False, action='store_true', help='whether to share weight in backbone')
        config = parser.parse_args()  # get command line parameters
        for key in config_dataset:
            if key in config and getattr(config, key) is not None: # if not set in command line, set it from .yml file
                continue
            setattr(config, key, config_dataset[key])

        if torch.cuda.is_available():
            config.nocuda = False
            config.device = 'cuda'
        else:
            config.nocuda = True
            config.device = 'cpu'
        return config


def setup_config_IIC(conf_yml, dataset='iic'):
    """Sets up the configs for infoseg
    Returns:
        ArgumentParser -- parser with arguments set.
    """
    with open(conf_yml, 'r') as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)
        config_dataset = config_dict[dataset]

        parser = argparse.ArgumentParser()
        config = parser.parse_args()  # get command line parameters
        for key in config_dataset:
            setattr(config, key, config_dataset[key])

        if torch.cuda.is_available():
            config.nocuda = False
            config.device = 'cuda'
        else:
            config.nocuda = True
            config.device = 'cpu'

        config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)
        assert (config.mode == "IID")
        assert (config.output_k_B == config.gt_k)
        config.output_k = config.output_k_B  # for eval code
        assert (config.output_k_A >= config.gt_k)  # sanity
        config.use_doersch_datasets = False
        config.eval_mode = "hung"

        if config.restart:
            config_name, dict_name = "configs.pickle", "latest.pytorch"  # checkpoint file

            reloaded_config_path = os.path.join(config.out_dir, config_name)
            print("Loading restarting configs from: %s" % reloaded_config_path)
            with open(reloaded_config_path, "rb") as config_f:
                config = pickle.load(config_f)
            assert (config.model_ind == config.model_ind)
            config.restart = True
            # copy over new num_epochs and lr schedule
            config.num_epochs = config.num_epochs
            config.lr_schedule = config.lr_schedule
        else:
            dict_name = False
            config.epoch_acc = []
            config.epoch_avg_subhead_acc = []
            config.epoch_stats = []

            config.epoch_loss_head_A = []
            config.epoch_loss_no_lamb_head_A = []

            config.epoch_loss_head_B = []
            config.epoch_loss_no_lamb_head_B = []
            print("Given configs: %s" % config_to_str(config))

        return config


def config_to_str(config):
    attrs = vars(config)
    string_val = "Config: -----\n"
    string_val += "\n".join("%s: %s" % item for item in attrs.items())
    string_val += "\n----------"
    return string_val

def config_infoseg_zy3():
    conf_yml = r'configs/train_infoseg.yml'
    conf_infoseg_zy3 = setup_config(conf_yml, dataset='zy3')
    print(uti.config_to_str(conf_infoseg_zy3))
    return conf_infoseg_zy3

def generate_config(yml_file, dataset_name):
    #print(yml_file)
    assert(os.path.exists(yml_file))
    # if 'infoseg' in yml_file:
    #     return setup_config(yml_file, dataset=dataset_name)
    if 'iic' in yml_file: # iic get different configuration
        return setup_config_IIC(yml_file, dataset=dataset_name)
    else:
        return setup_config(yml_file, dataset=dataset_name)


if __name__=='__main__':
    #show an example of usage.
    conf_yml = r'../configs/train_infoseg.yml'
    conf_infoseg_zy3 = setup_config(conf_yml)
    print(uti.config_to_str(conf_infoseg_zy3))
    exit(0)
