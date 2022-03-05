import random
import numpy as np
import cv2
from tensorflow.python.keras.utils.data_utils import Sequence
import secrets as s
import math as m
from tensorflow.python.keras.utils import to_categorical


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, df, batch_size=32, n_branches=7, n_classes=None, shuffle=True, ratios=None, image_width=None,
                 image_height=None):

        """Initialization"""
        self.batch_size = batch_size
        self.df = df.copy()
        self.n = len(self.df)
        self.n_branches = n_branches
        if n_classes is None:
            self.n_classes = [4] * n_branches
        else:
            self.n_classes = n_classes
        self.shuffle = shuffle
        self.ratios = ratios
        self.image_width = image_width,
        self.image_height = image_height,
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(self.n // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = np.array([])
        if self.ratios is None:
            indexes = self.df[index * self.batch_size:(index + 1) * self.batch_size].index.values
        else:
            count, cols = self.batch_size / sum(self.ratios), list(self.df.columns)
            cols.remove('img_name')
            cols.remove('img_path')
            for i in range(0, self.n_branches):
                class_indices, proportion_count = list(self.df[self.df[cols[i]] > 0].index.values), int(count *
                                                                                                        self.ratios[i])
                if index * proportion_count >= len(class_indices):
                    """For rare class the index value will restart when all the samples in a class are used once"""
                    temp_index = index % m.ceil(len(class_indices) / proportion_count)
                    selected_index = np.array(
                        class_indices[(temp_index * proportion_count):((temp_index + 1) * proportion_count)],
                        dtype=np.int)
                    indexes = np.concatenate([indexes, selected_index]).astype('int')
                elif index * proportion_count < len(class_indices) & (index + 1) * proportion_count > \
                        len(class_indices):
                    """Select last few indices of the rare class and then select random to fill the class proportion"""
                    selected_index = np.array(class_indices[(index * proportion_count):], dtype=np.int)
                    indexes = np.concatenate([indexes, selected_index]).astype('int')
                    """For imbalanced data when each sample is traversed once so select the random proportional from 
                    the same class"""
                    diff = (index + 1) * proportion_count - (index * proportion_count) - len(selected_index)
                    remaining_indices = list(set(class_indices) - set(selected_index))
                    selected_index = np.array(random.choices(remaining_indices, k=diff), dtype=np.int)
                    indexes = np.concatenate([indexes, selected_index]).astype('int')
                else:
                    """Normal case select the proportion of data from each class"""
                    selected_index = np.array(class_indices[(index * proportion_count):((index + 1) * proportion_count)]
                                              , dtype=np.int)
                    indexes = np.concatenate([indexes, selected_index]).astype('int')

            """When batch size is not completely divisible by the sum of input ratios"""
            i, diff = 0, self.batch_size - indexes.shape[0]
            while i < diff:
                temp_i = i % self.n_branches
                class_indices, proportion_count = list(self.df[self.df[cols[temp_i]] > 0].index.values), \
                                                  int(count * self.ratios[temp_i])
                unselected_index = class_indices[:(index * proportion_count)] + class_indices[
                                                                                ((index + 1) * proportion_count):]
                indexes = np.append(indexes, s.choice(unselected_index))
                i += 1

        """Generate Data"""
        x, y = self.__data_generation(indexes)

        """Normalizing Data"""
        x = x / 255.

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        x = []
        y = [[] for _ in range(self.n_branches)]
        for i, ID in enumerate(indexes):
            img = cv2.resize(cv2.cvtColor(cv2.imread(self.df.iloc[ID, 1], 1), cv2.COLOR_BGR2RGB),
                             (self.image_width[0], self.image_height[0]))
            x.append(img)
            [sub.append(to_categorical(self.df.iloc[ID, y.index(sub)+2], self.n_classes[y.index(sub)])) for sub in y]
        return np.array(x), y