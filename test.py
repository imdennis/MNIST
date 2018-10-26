from PIL import Image
import numpy as np
import random

def test_data(str):
    bmp_image = Image.open(str)
    test = []
    line_horizon = ((sum(bmp_image.getpixel((i_horizon, i_vertical))))/3/256 for i_vertical in range(bmp_image.height) for i_horizon in range(bmp_image.width))
    for x in range(bmp_image.height):
        for y in range(bmp_image.width):
            value = float(next(line_horizon))
            test.append(value)

    bmp_image.close()
    return np.array([test])