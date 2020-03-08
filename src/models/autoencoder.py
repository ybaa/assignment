import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, Input, MaxPooling2D

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, input):
        encoded = self.encoder(input)
        reconstructed = self.decoder(encoded)
        return reconstructed

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(32, 4, activation='relu', padding='same')
        self.b1 = BatchNormalization()
        self.pool = MaxPooling2D(2)
        self.conv2 = Conv2D(64, 4, activation='relu', padding='same')
        self.b2 = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.b2(x)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(32, 4, activation='relu', padding='same')
        self.b1 = BatchNormalization()
        self.up1 = UpSampling2D(2)
        # self.conv2 = Conv2D(32, 3, activation='relu', padding='same')
        # self.b2 = BatchNormalization()
        # self.up2 = UpSampling2D(2)
        self.decode = Conv2D(3, 4, activation='sigmoid', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.up1(x)
        # x = self.conv2(x)
        # x = self.b2(x)
        # x = self.up2(x)
        x = self.decode(x)
        return x

