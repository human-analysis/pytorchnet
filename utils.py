# utils.py

import os
import copy
import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import compare_ssim as ssim

def readtextfile(filename):
    with open(filename) as f:
        content = f.readlines()
    f.close()
    return content

def writetextfile(data, filename):
    with open(filename, 'w') as f:
        f.writelines(data)
    f.close()

def delete_file(filename):
    if os.path.isfile(filename) == True:
        os.remove(filename)