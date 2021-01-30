import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from unet import UNet


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


def preprocess_data(data_dir, size):
    file_names = np.array(os.listdir(data_dir))

    # might need tweaking if file names or sorting differs
    image_file_names = file_names[1::4]
    mask_file_names = file_names[3::4]

    # combine masks of identical input image to combined masks
    prev_image_array = None
    prev_mask_arrays = []

    for index, (image_file_name, mask_file_name) in tqdm(enumerate(zip(image_file_names, mask_file_names))):

        # resize images
        image = Image.open(os.path.join(data_dir, image_file_name))
        image_resized = image.resize(size, Image.ANTIALIAS)
        image_resized_array = np.asarray(image_resized)
        
        # resize, grayscale and binary threshold masks
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

            out_dir_image = "data/images"
            out_dir_mask = "data/masks"

            if not os.path.exists(out_dir_image):
                os.makedirs(out_dir_image)
            if not os.path.exists(out_dir_mask):
                os.makedirs(out_dir_mask)

            image.save(os.path.join(out_dir_image,"{0}.png".format(index)))
            mask.save(os.path.join(out_dir_mask,"{0}.png".format(index)))

            prev_image_array = image_resized_array.copy()
            prev_mask_arrays = [mask_resized_grayscale_binary_array.copy()]


def get_data_size(dir):
    return len(os.listdir(os.path.join(dir, 'images', 'img')))


def get_data(dir):
    file_names = np.array(os.listdir(dir))
    data = np.asarray([np.asarray(Image.open(os.path.join(dir, file_name))) for file_name in file_names])

    # grayscale images only have 3 dims(num_images, height, width), but DataGenerator requires 4 dims(num_images, height, width, color_channels)
    if len(data.shape) < 4:
        data = np.expand_dims(data, 3)
    return data


def get_data_generators(batch_size, dir):

    # set parameters for data augmentation
    data_gen_args = dict(featurewise_center=True,
                        featurewise_std_normalization=True,
                        rotation_range=90,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    images = get_data(os.path.join(dir, 'images', 'img'))
    masks = get_data(os.path.join(dir, 'masks', 'img'))

    # set seed to transform images and masks equally
    seed = 1

    image_datagen.fit(images, seed=seed)
    mask_datagen.fit(masks, seed=seed)

    image_generator = image_datagen.flow(images, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(masks, batch_size=batch_size, seed=seed)

    return image_generator, mask_generator

def display(display_list, batch):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.savefig(f"plot{batch}.png")



def main():
    DATA_DIR = "data/"
    SIZE = 256, 144

    # preprocess inputs, combine masks and transform to grayscale binary
    # preprocess_data(DATA_DIR, SIZE)

    BATCH_SIZE = 25

    image_generator, mask_generator = get_data_generators(BATCH_SIZE, DATA_DIR)

    model = UNet(SIZE)

    # TODO find correct metric and loss
    model.compile(optimizer='adam', metrics=['mse'], loss='binary_crossentropy')

    keras.utils.plot_model(model, show_shapes=True)

    DATA_SIZE = 250#get_data_size(DATA_DIR)

    print(f"nr. of images: {DATA_SIZE}")
    EPOCHS = 25

    test_img = np.asarray(Image.open("data/images/999.png"))
    test_mask = np.asarray(Image.open("data/masks/999.png"))

    # TODO validation split

    histories = []

    # train
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        BATCH_ITERATION = 1
        for image_batch, mask_batch in zip(image_generator, mask_generator):

            # break on overflow images if DATA_SIZE % BATCH_SIZE != 0
            if BATCH_ITERATION * BATCH_SIZE > DATA_SIZE: break
            
            histories.append(model.fit(image_batch, mask_batch))

            BATCH_ITERATION += 1

        test_pred = model.predict(np.expand_dims(test_img, axis=0))

        display([test_img, np.expand_dims(test_mask, 2), test_pred[0]], epoch)
    
    # TODO test, test split
    # TODO visualize results


if __name__ == "__main__":
    print("GPU") if len(tf.config.experimental.list_physical_devices('GPU')) > 0 else print("CPU")
    main()
