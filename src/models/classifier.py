import numpy as np
import tensorflow as tf
from src.models.autoencoder import Encoder
from tensorflow.keras.layers import Dense, Flatten

class ClassificationAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ClassificationAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()

    def call(self, input):
        encoded = self.encoder(input)
        out = self.classifier(encoded)
        return out

class Classifier(tf.keras.layers.Layer):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flat = Flatten()
        self.dense = Dense(128, activation='relu')
        self.out = Dense(10, activation='softmax')

    def call(self,input):
        x = self.flat(input)
        x = self.dense(x)
        x = self.out(x)
        return x
