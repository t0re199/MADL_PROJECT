import os

import numpy as np
from PIL import Image

from Constants import IMAGE_DIRECTORY_PATH, SKIP_EXTENSION


def mark_to_skip(image_name, directory=IMAGE_DIRECTORY_PATH):
    old_name = os.path.join(directory, image_name)
    new_name = os.path.join(directory, image_name + SKIP_EXTENSION)
    os.rename(old_name, new_name)


def write_image_to_file(image, image_name, directory=IMAGE_DIRECTORY_PATH):
    file = os.path.join(directory, image_name)
    image = Image.fromarray(image.astype(np.uint8))
    image.save(file)
