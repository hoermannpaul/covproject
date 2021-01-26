import os

import numpy as np
from PIL import Image, ImageOps

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

from tqdm import tqdm

from unet import UNet

size = 320, 180

# read data into X(original images) and y(segmented images)
# resize to largest image found (all share the same aspect reation)
# convert y to grayscale
# return numpy arrays
def preprocess_data(data_dir):
    file_names = np.array(os.listdir(data_dir)[1:])

    X_file_names = file_names[1::4]
    y_file_names = file_names[3::4] 

    for index, (X_file_name, y_file_name) in tqdm(enumerate(zip(X_file_names, y_file_names))):

        X_image = Image.open(os.path.join(data_dir, X_file_name))
        X_image_resized = X_image.resize(size, Image.ANTIALIAS)
        X_image_resized.save("data/images/{0}.png".format(index))

        y_image = Image.open(os.path.join(data_dir, y_file_name))
        y_image_resized = y_image.resize(size, Image.ANTIALIAS)
        y_image_resized_grayscale = ImageOps.grayscale(y_image_resized)
        y_image_resized_grayscale_binary = y_image_resized_grayscale.point(lambda p: p > 0 and 255) 
        y_image_resized_grayscale_binary.save("data/masks/{0}.png".format(index))


def get_train_generator():
    seed = 909

    # ["constant", "nearest", "reflect", "wrap"]
    image_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1],
        zoom_range=[0.5, 1],
        fill_mode="reflect")

    mask_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1],
        zoom_range=[0.5, 1],
        fill_mode="reflect")

    image_generator =image_datagen.flow_from_directory("data/images", class_mode=None, seed=seed)
    mask_generator = mask_datagen.flow_from_directory("data/masks", class_mode=None, seed=seed)
    
    train_generator = zip(image_generator, mask_generator)

    return train_generator


def main():
    DATA_DIR = "data/"
    # preprocess_data(DATA_DIR)

    train_generator = get_train_generator()
    model = UNet(size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    # print(model.summary())


if __name__ == "__main__":
    main()
