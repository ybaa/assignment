import argparse
import sys,os
sys.path.append(os.path.realpath('.'))

from src.helpers.cifarHelper import CifarHelper
from src.data.cifarLoader import load_data_cifar10
from src.configs.configParser import parse_config
from src.models.autoencoder import Autoencoder
from src.models.classifier import ClassificationAutoencoder

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import datetime
import tensorflow as tf

from tensorflow.keras.metrics import Accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to configuration file')
    parser.add_argument('model', type=str, help='Model to train')
    args = parser.parse_args()

    print('args: ', args)
    config = parse_config(args.config_path)
    
    meta, data = load_data_cifar10(config.data.cifar_10_path)

    cifar_helper = CifarHelper(data, meta, config)
    cifar_helper.set_up_images()
    
    if args.model == 'classifier_and_encoder' or args.model == 'classifier':
        full_model = ClassificationAutoencoder()
        full_model.compile(loss=config.model.classifier.loss,
                        optimizer= config.model.classifier.optimizer,
                        metrics=[Accuracy()])

        log_dir="models/logs/" + config.model.classifier.tag + "/" 
        
        if config.model.classifier.restore:
            path = tf.train.latest_checkpoint(log_dir)
            full_model.load_weights(path)

        predictions = full_model.predict(cifar_helper.test_images[:15])

        n = 15
        plt.figure(figsize=(20, 4))
        for i in range(1,n):
            print(cifar_helper.get_name_from_one_hot(predictions[i]))
            # display original
            ax = plt.subplot(2, n, i)
            plt.imshow(cifar_helper.test_images[i].reshape(32,32,3))
            title = cifar_helper.get_name_from_one_hot(predictions[i]) + '\n' + cifar_helper.get_name_from_one_hot(cifar_helper.test_labels[i])
            plt.title(title)
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()

    elif args.model == 'autoencoder':
        autoencoder = Autoencoder()
        autoencoder.compile(loss=config.model.autoencoder.loss, 
                            optimizer=config.model.autoencoder.optimizer)

        log_dir="models/logs/" + config.model.autoencoder.tag + "/" 

        if config.model.autoencoder.restore:
            path = tf.train.latest_checkpoint(log_dir)
            autoencoder.load_weights(path)

        predictions = autoencoder.predict(cifar_helper.test_images[:15])
        n = 15
        plt.figure(figsize=(20, 4))
        for i in range(1,n):
            # display original
            ax = plt.subplot(2, n, i)            
            plt.imshow(cifar_helper.test_images[i].reshape(32,32,3))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(predictions[i].reshape(32,32,3))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    else:
        print('unknown model')