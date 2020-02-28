import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random

class CifarHelper:

    def __init__(self, all_batches, batches_meta, config):
        self.next_batch_start_index = 0

        self.all_batches = all_batches

        self.training_images = []
        self.training_labels = []

        self.test_images = []
        self.test_labels = []

        self.batches_meta = batches_meta

        self.config = config
        
    def set_up_images(self):

        images = np.vstack([d[b"data"] for d in self.all_batches])
        images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1) / 255

        labels = np.hstack([d[b"labels"] for d in self.all_batches])

        self._group_by_category(images, labels)
        self._shuffle_all(2020)
        
        self.training_labels = self._one_hot_encode(self.training_labels)
        self.test_labels = self._one_hot_encode(self.test_labels)

        print('train len: ', len(self.training_images))
        print('test len: ', len(self.test_images))
      

    def next_batch(self, batch_size):
        x = self.training_images[self.next_batch_start_index:self.next_batch_start_index + batch_size].reshape(batch_size, 32, 32, 3)
        y = self.training_labels[self.next_batch_start_index:self.next_batch_start_index + batch_size]
        self.next_batch_start_index = (self.next_batch_start_index + batch_size) % len(self.training_images)
        return x, y

    def _one_hot_encode(self, vec):
        n = len(vec)
        out = np.zeros((n, 10))
        out[range(n), vec] = 1
        return out

    def _get_category_name(self, index):
        return self.batches_meta[b'label_names'][index].decode()

    def _group_by_category(self, images, labels):
        grouped = dict()
        for i, label in enumerate(labels):
            if label in grouped.keys():
                grouped[label].append(images[i])
            else:
                grouped[label] = [images[i]]

        for category, images in grouped.items():
            test_size = getattr(self.config.data.split_test, self._get_category_name(category))
            y = np.full(shape=len(images), fill_value=category)

            X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=test_size, random_state=42)
            
            self.training_images += X_train
            self.test_images += X_test
            self.training_labels += list(y_train)
            self.test_labels += list(y_test)
        
    def _shuffle_all(self,seed):
        random.Random(seed).shuffle(self.training_images)
        random.Random(seed).shuffle(self.test_images)
        random.Random(seed).shuffle(self.training_labels)
        random.Random(seed).shuffle(self.test_labels)