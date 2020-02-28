import pickle
import glob

import sys,os
sys.path.append(os.path.realpath('.'))

from src.configs.configParser import parse_config

def load_data_cifar10(dir_path):
    data = []

    filenames = sorted(glob.glob(dir_path + '*batch*'))

    for dir in filenames:
        data.append(_unpickle(dir))

    return data[0], data[1:]
 
def _unpickle(file):
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


if __name__ == "__main__":
    cfg_path = '/run/media/ybaa/Data/assignment/src/configs/assignment.yml'
    cfg = parse_config(cfg_path)
    load_data_cifar10(cfg.data.cifar_10_path)
    pass