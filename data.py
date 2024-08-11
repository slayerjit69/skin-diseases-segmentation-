import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from glob import glob
import tensorflow as tf
import cv2

def load_data(paths):
    train_images = sorted(glob(os.path.join(paths, 'train/*')))
    train_masks = sorted(glob(os.path.join(paths, 'train_GT/*')))

    valid_images = sorted(glob(os.path.join(paths, 'valid/*')))
    valid_masks = sorted(glob(os.path.join(paths, 'valid_GT/*')))

    test_images = sorted(glob(os.path.join(paths, 'test/*')))
    test_masks = sorted(glob(os.path.join(paths, 'test_GT/*')))

    return (train_images, train_masks), (valid_images, valid_masks), (test_images, test_masks)

def read_image(paths):
    try:
        paths = paths.decode()
        x = cv2.imread(paths, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (256, 256))
        x = x / 255.0
        #print(f"Read image: {paths}, shape: {x.shape}, min: {x.min()}, max: {x.max()}")
        return x
    except Exception as e:
        print("Error reading image:", e)
        return None

def read_mask(paths):
    try:
        paths = paths.decode()
        x = cv2.imread(paths, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (256, 256))
        x = x / 255.0
        x = np.expand_dims(x, axis=-1)
        #print(f"Read mask: {paths}, shape: {x.shape}, unique values: {np.unique(x)}")
        return x
    except Exception as e:
        print("Error reading mask:", e)
        return None

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

if __name__ == "__main__":
    try:
        path = r"D:\Python\melanoma-skin-cancer-image-segmentation-master\dataset"
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

        print("Length of test_x:", len(test_x))
        print("Length of test_y:", len(test_y))

        ds = tf_dataset(train_x, train_y)
        for x, y in ds:
            print("Batch shape - input:", x.shape, "output:", y.shape)
            break
    except Exception as e:
        print("Error:", e)
