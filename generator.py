import os
import sys
import random
import numpy as np
from skimage import io
from skimage.transform import resize
from tensorflow import keras
from keras.models import Sequential
from skimage import data,exposure
from skimage import io, transform

class generator(keras.utils.Sequence):
    def __init__(self, folder, filenames, batch_size=32, image_size=256, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def __load__(self, filename):
        pass

    def __loadpredict__(self, filename):
        if os.path.exists(os.path.join(self.folder, filename)):
            img = io.imread(os.path.join(self.folder, filename))
            # resize image
            img = resize(img, (self.image_size, self.image_size), mode='reflect')
            img = exposure.equalize_hist(img)
            img = (img - np.mean(img)) / np.std(img)
            # add trailing channel dimension
            img = np.expand_dims(img, -1)
            return img
        else:
            pass

    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        else:
            pass

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    # 使用内建函数 len()，需要实现__len__()方法
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)