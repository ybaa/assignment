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
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.optimizers import Adam


# TODO: organize this file somehow, theres a lot of things with common logic
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
    
    # train full model from the beginning including encoder
    if args.model == 'classifier_and_encoder':
        log_dir="models/logs/" + config.model.classifier.tag + "/" 
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir + 'cp-{epoch:04d}.ckpt',
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        period=config.model.classifier.ckpt_period_save)
        
        optimizer = Adam(learning_rate=config.model.classifier.learning_rate)
        full_model = ClassificationAutoencoder()
        full_model.compile(loss=config.model.classifier.loss,
                        optimizer= optimizer,
                        metrics=[CategoricalAccuracy(),
                                 F1Score(average='macro', num_classes=10)])
        
        if config.model.classifier.restore:
            path = tf.train.latest_checkpoint(log_dir)
            full_model.load_weights(path)

        full_model_trained = full_model.fit(cifar_helper.training_images, cifar_helper.training_labels, 
                                            batch_size=config.model.classifier.batch_size,
                                            epochs=config.model.classifier.epochs,
                                            verbose=1,
                                            initial_epoch=config.model.classifier.initial_epoch,
                                            validation_data=(cifar_helper.test_images, cifar_helper.test_labels),
                                            callbacks=[tensorboard_callback,
                                                       cp_callback])
        
    # train only autoencoder
    elif args.model == 'autoencoder':

        log_dir="models/logs/" + config.model.autoencoder.tag + "/" 
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir + 'cp-{epoch:04d}.ckpt',
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        period=config.model.autoencoder.ckpt_period_save)
        autoencoder = Autoencoder()
        autoencoder.compile(loss=config.model.autoencoder.loss, 
                            optimizer=config.model.autoencoder.optimizer)

        if config.model.autoencoder.restore:
            path = tf.train.latest_checkpoint(log_dir)
            autoencoder.load_weights(path)

        autoencoder_train = autoencoder.fit(cifar_helper.training_images, 
                                            cifar_helper.training_images, 
                                            batch_size=config.model.autoencoder.batch_size,
                                            epochs=config.model.autoencoder.epochs,
                                            verbose=1,
                                            validation_data=(cifar_helper.test_images, cifar_helper.test_images),
                                            callbacks=[tensorboard_callback,
                                                       cp_callback])

    # train only classifier
    elif args.model == 'classifier':
        ca = ClassificationAutoencoder()
        optimizer = Adam(learning_rate=config.model.classifier.learning_rate)
        ca.compile(loss=config.model.classifier.loss, 
                   optimizer=optimizer,
                   metrics=[CategoricalAccuracy(), 
                            F1Score(average='macro', num_classes=10)])

        ca_log_dir="models/logs/" + config.model.classifier.tag + "/" 
        autoencoder_log_dir="models/logs/" + config.model.autoencoder.tag + "/" 


        if config.model.classifier.restore:
            path = tf.train.latest_checkpoint(ca_log_dir)
            ca.load_weights(path)

        # restore autoencoder part
        autoencoder = Autoencoder()
        autoencoder.compile(loss=config.model.autoencoder.loss, 
                            optimizer=config.model.autoencoder.optimizer)
        
        autoencoder.build((None,32,32,3))

        autoencoder_path = tf.train.latest_checkpoint(autoencoder_log_dir)
        autoencoder.load_weights(autoencoder_path)

        weights = autoencoder.get_weights
        ca.build((None,32,32,3))
        
        ca.layers[0].set_weights(autoencoder.layers[0].get_weights())
        ca.layers[0].trainable=False

        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=ca_log_dir, histogram_freq=1, profile_batch=0)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ca_log_dir + 'cp-{epoch:04d}.ckpt',
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        period=config.model.classifier.ckpt_period_save)

        ca_trained = ca.fit(cifar_helper.training_images, cifar_helper.training_labels, 
                            batch_size=config.model.classifier.batch_size,
                            epochs=config.model.classifier.epochs,
                            verbose=1,
                            initial_epoch=config.model.classifier.initial_epoch,
                            validation_data=(cifar_helper.test_images, cifar_helper.test_labels),
                            callbacks=[tensorboard_callback,
                                       cp_callback])


    else:
        print('unknown model')
        
