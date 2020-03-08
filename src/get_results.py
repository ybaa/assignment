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

import datetime
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy

import seaborn as sn
import pandas as pd
import tensorflow_addons as tfa
from tensorflow_addons.metrics import MultiLabelConfusionMatrix
from sklearn.metrics import accuracy_score, f1_score


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
                        metrics=[Accuracy(), CategoricalAccuracy()])

        

        log_dir="models/logs/" + config.model.classifier.tag + "/" 
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
        
        if config.model.classifier.restore:
            path = tf.train.latest_checkpoint(log_dir)
            full_model.load_weights(path)

        # ev = full_model.evaluate(cifar_helper.test_images, cifar_helper.test_labels)

        predictions = full_model.predict(cifar_helper.test_images)

        rev_test_labels = cifar_helper.reverse_one_hot(cifar_helper.test_labels)
        rev_predictions = cifar_helper.reverse_one_hot(predictions)
        
        acc = accuracy_score(rev_test_labels, rev_predictions)        
        f1 = f1_score(rev_test_labels, rev_predictions,average='macro')
        print('acc: ', acc, '| f1: ', f1)
        
        output = tf.math.confusion_matrix(
            rev_test_labels, rev_predictions, num_classes=10, dtype=tf.dtypes.int32
        )
        
        cm = output.numpy()
        
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize = (12,9))
        sn.heatmap(cm, 
                   annot=True, 
                #    fmt='2d', 
                   fmt='0.2f', 
                   xticklabels=cifar_helper.batches_meta[b'label_names'],
                   yticklabels=cifar_helper.batches_meta[b'label_names'],
                   cmap = plt.cm.Blues)
        plt.title(config.model.classifier.tag)
        plt.show()


    else:
        print('unknown model')