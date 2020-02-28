import argparse
import sys,os
sys.path.append(os.path.realpath('.'))

from src.helpers.cifarHelper import CifarHelper
from src.data.cifarLoader import load_data_cifar10
from src.configs.configParser import parse_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to configuration file')
    args = parser.parse_args()

    print('args: ', args)
    config = parse_config(args.config_path)
    
    meta, data = load_data_cifar10(config.data.cifar_10_path)

    cifar_helper = CifarHelper(data, meta, config)
    cifar_helper.set_up_images()
   