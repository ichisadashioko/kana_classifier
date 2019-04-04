from __future__ import absolute_import, division, print_function

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


def compile_etl_datasets():
    data_root = pathlib.Path('./etlcb_01_datasets/')
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]

    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index)
                         for index, name in enumerate(label_names))

    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    def process_image(img):
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize_images(img, [160, 160])

        img /= 255.0
        return img

    def load_and_process_image(path):
        img = tf.read_file(path)
        return process_image(img)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    img_ds = path_ds.map(load_and_process_image, num_parallel_calls=AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_image_labels, tf.int64))

    return img_ds, label_ds
