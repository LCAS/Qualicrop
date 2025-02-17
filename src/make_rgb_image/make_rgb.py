# Started by Usman Zahidi (uzahidi@lincoln.ac.uk)
# makes a Narrow-band RGB image from hyperspectral cube

# arguments:
# hsi_image : hyperspectral cube
# lambda_ : wavelength vector in micrometer
# brightness: brightness of the developed image

import numpy as np

def find_rgb_bands(lambda_):
    # make function robust to various scalings of lambda
    while not (lambda_[0] > 0 and lambda_[0] < 1):
        lambda_ = lambda_ / 10
    if lambda_[0] > 1:
        lambda_ = lambda_ / 1000

    # define wavelengths of colours
    red = 0.6329  # 0.65
    green = 0.5510  # 0.510
    blue = 0.454528  # 0.475

    S = lambda_.shape

    B = np.max(np.where(lambda_ <= blue)[0]) if np.any(lambda_ <= blue) else 0
    R = np.max(np.where(lambda_ <= red)[0]) if np.any(lambda_ <= red) else S[1] - 1
    G = np.max(np.where(lambda_ <= green)[0]) if np.any(lambda_ <= green) else 1

    out = [R, G, B]
    return out

def make_rgb(hsi_image, lambda_, brightness=1.25):
    # find rgb bands
    out = find_rgb_bands(lambda_)

    # create unnormalized RGB image
    rgb_image = np.concatenate((hsi_image[:, :, out[0:1]],
                                 hsi_image[:, :, out[1:2]],
                                 hsi_image[:, :, out[2:3]]), axis=2)
    # normalize rgb image
    for i in range(3):
        rgb_image[:, :, i] = rgb_image[:, :, i] - np.min(rgb_image[:, :, i])
        rgb_image[:, :, i] = (brightness * rgb_image[:, :, i]) / np.max(rgb_image[:, :, i])

    return rgb_image
