import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from unet import UNet


size = 256, 144


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def preprocess_data(data_dir):
    file_names = np.array(os.listdir(data_dir)[:])

    image_file_names = file_names[1::4]
    mask_file_names = file_names[3::4]

    prev_image_array = None
    prev_mask_arrays = []

    for index, (image_file_name, mask_file_name) in tqdm(enumerate(zip(image_file_names, mask_file_names))):

        image = Image.open(os.path.join(data_dir, image_file_name))
        image_resized = image.resize(size, Image.ANTIALIAS)
        image_resized_array = np.asarray(image_resized)

        mask = Image.open(os.path.join(data_dir, mask_file_name))
        mask_resized = mask.resize(size, Image.ANTIALIAS)
        mask_resized_grayscale = ImageOps.grayscale(mask_resized)
        mask_resized_grayscale_binary = mask_resized_grayscale.point(lambda p: p > 0 and 255)
        mask_resized_grayscale_binary_array = np.asarray(mask_resized_grayscale_binary)

        if (prev_image_array is None) or (np.array_equal(image_resized_array, prev_image_array)):
            if prev_image_array is None:
                prev_image_array = image_resized_array.copy()
            prev_mask_arrays.append(mask_resized_grayscale_binary_array.copy())
        else:
            combined_mask = prev_mask_arrays[0].copy()
            for i in range(1, len(prev_mask_arrays)):
                combined_mask = np.add(combined_mask, prev_mask_arrays[i])

            combined_mask[combined_mask > 255] = 255

            image = Image.fromarray(prev_image_array)
            mask = Image.fromarray(combined_mask)

            image.save("data/images/{0}.png".format(index))
            mask.save("data/masks/{0}.png".format(index))

            prev_image_array = image_resized_array.copy()
            prev_mask_arrays = [mask_resized_grayscale_binary_array.copy()]


def get_train_generator():

    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                        featurewise_std_normalization=True,
                        rotation_range=90,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    # image_datagen.fit(images, augment=True, seed=seed)
    # mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        'data/images',
        class_mode=None,
        classes=None,
        target_size=size,
        color_mode='rgb',
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'data/masks',
        class_mode=None,
        classes=None,
        target_size=size,
        color_mode='grayscale',
        seed=seed)

    return zip(image_generator, mask_generator)
    # seed = 909

    # # ["constant", "nearest", "reflect", "wrap"]
    # image_datagen = ImageDataGenerator(
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     rotation_range=90,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     brightness_range=[0.5, 1],
    #     zoom_range=[0.5, 1],
    #     fill_mode="reflect")

    # mask_datagen = ImageDataGenerator(
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     rotation_range=90,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     brightness_range=[0.5, 1],
    #     zoom_range=[0.5, 1],
    #     fill_mode="reflect")

    # image_generator = image_datagen.flow_from_directory("data/images/", target_size=size, class_mode=None, seed=seed)
    # mask_generator = mask_datagen.flow_from_directory("data/masks/", target_size=size, color_mode='grayscale', class_mode=None, seed=seed)

    # return image_generator, mask_generator


def main():
    DATA_DIR = "data/"
    preprocess_data(DATA_DIR)



if __name__ == "__main__":
    print("GPU") if len(tf.config.experimental.list_physical_devices('GPU')) > 0 else print("CPU")
    main()
