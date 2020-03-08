import numpy as np
import tensorflow as tf
from src.models.autoencoder import Encoder
from tensorflow.keras.layers import Dense, Flatten, Dropout

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
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(64, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.out = Dense(10, activation='softmax')

    def call(self,input):
        x = self.flat(input)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.out(x)
        return x
