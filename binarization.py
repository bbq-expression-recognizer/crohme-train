import numpy as np
import math
from scipy import ndimage
from PIL import Image
from PIL import ImageOps

def to_binary(gray_image):
    """ Transform grayscale image into binary image
    Use local thresholding technique
    returns numpy uint8 2d array
    """
    width,height = gray_image.size
    raw_array = np.asarray(gray_image).copy()

    gray_image = Image.fromarray(raw_array, 'L')

    # contrast image
    gray_image = ImageOps.autocontrast(gray_image)
    raw_array = np.asarray(gray_image).copy()

    # sum up nearby pixels.
    near_uniform_array = ndimage.filters.uniform_filter(
        raw_array.astype(np.int16),
        size=max(2, width / 300, height / 300))

    # calculate regional mean for each point
    uniform_array = ndimage.filters.uniform_filter(
        raw_array.astype(np.int16),
        size=max(10, width / 50, height / 50))

    # estimate threshold
    thres = np.min(near_uniform_array - uniform_array) / 6

    # difference to mean of its region
    mask = (near_uniform_array < (uniform_array + thres))

    raw_array[mask] = 0
    raw_array[~mask] = 255

    # save binary image for debugging
    #Image.fromarray(raw_array, 'L').save('binary.png')
    return raw_array


def rotated_angle(binary_image):
    # remove noise dots
    binary_image = ndimage.grey_dilation(binary_image, size=(3,3))

    dark_positions = np.where(binary_image == 0)

    def get_height_dev(angle):
        dist = dark_positions[1] * math.sin(angle) - dark_positions[0] * math.cos(angle)
        return np.std(dist)

    low = -math.pi/6
    high = math.pi/6

    best_guess = 0
    best_guess_val = get_height_dev(best_guess)
    for magic in xrange(0,20):
        mid1 = (low*2 + high) / 3
        mid2 = (low + high*2) / 3

        width1 = get_height_dev(mid1)
        width2 = get_height_dev(mid2)
        if width1 < width2:
            high = mid2
        else:
            low = mid1
        best_guess_val, best_guess = min((best_guess_val, best_guess), (width1, mid1))
        best_guess_val, best_guess = min((best_guess_val, best_guess), (width2, mid2))

    return best_guess

def rotate_image(binary_image, angle):
	binary_image = ndimage.rotate(binary_image, angle * 180 / math.pi,
		mode='nearest', order=5, prefilter=False)
	binary_image[binary_image<128] = 0
	binary_image[binary_image>=128] = 255
	return binary_image

