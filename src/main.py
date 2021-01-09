import os

import numpy as np
from PIL import Image, ImageOps


# read data into X(original images) and y(segmented images)
# resize to largest image found (all share the same aspect reation)
# convert y to grayscale
# return numpy arrays
def get_data(data_dir):
    file_names = np.array(os.listdir(data_dir)[1:])

    X_file_names = file_names[1::4]
    y_file_names = file_names[3::4]

    X = []
    y = []

    size = 1280, 720

    for X_file_name, y_file_name in zip(X_file_names, y_file_names):

        X_image = Image.open(os.path.join(data_dir, X_file_name))
        X_image_resized = X_image.resize(size, Image.ANTIALIAS)
        X.append(np.array(X_image_resized))

        y_image = Image.open(os.path.join(data_dir, y_file_name))
        y_image_resized = y_image.resize(size, Image.ANTIALIAS)
        y_image_resized_grayscale = ImageOps.grayscale(y_image_resized)
        y.append(np.array(y_image_resized_grayscale))

    return np.array(X), np.array(y)


def main():
    DATA_DIR = "../data/"
    X, y = get_data(DATA_DIR)
    print(X.shape)
    print(y.shape)


if __name__ == "__main__":
    main()