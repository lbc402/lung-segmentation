import os
import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from generator import generator
from skimage import io
from keras.models import load_model
from skimage import measure
from skimage import morphology
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

def load_image(folder):
    filenames = os.listdir(folder)
    print(len(filenames))
    return filenames

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

if __name__ == '__main__':

    DATA_DIR = './testImage'
    COUNT = 0

    test_filenames = load_image(DATA_DIR)
    random.shuffle(test_filenames)
    imageCount = len(test_filenames)

    UNet = load_model('./trained_model.hdf5')

    test_gen = generator(DATA_DIR, test_filenames, batch_size=1, image_size=256, shuffle=False, predict=True)
    for imgs, filenames in tqdm(test_gen):
        COUNT+=1
        print(len(filenames))
        preds = UNet.predict(imgs)

        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            pred = resize(pred, (1024, 1024), mode='reflect')
            comp = pred[:, :, 0] > 0.5
            comp = remove_small_regions(comp, 0.02 * np.prod(1024))
            # comp = measure.label(comp)
            comp = comp * 255
            src = os.path.join('./resultImage', filename[:36] + '_mask.png')
            io.imsave(src, comp)

        if COUNT >= imageCount:
            break


