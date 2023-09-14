import numpy as np
from PIL import Image


def get_dark_prior(img_arr, r=3):
    min_chn = np.pad(img_arr.min(axis=2), (r, r), constant_values=img_arr.max())
    dark_prior_view = np.lib.stride_tricks.as_strided(
        min_chn,
        (*img_arr.shape[:2], r * 2 + 1, r * 2 + 1),
        strides=min_chn.strides * 2
    )
    return dark_prior_view.min(axis=(2, 3))


def get_transmission(img_arr, A, r=3, o=0.95):
    img_arr = img_arr / A
    min_chn = np.pad(img_arr.min(axis=2), (r, r), constant_values=img_arr.max())
    tran_view = np.lib.stride_tricks.as_strided(
        min_chn,
        (*img_arr.shape[:2], r * 2 + 1, r * 2 + 1),
        strides=min_chn.strides * 2
    )
    return 1 - o * tran_view.min(axis=(2, 3))


def get_air_light(img_arr, dark_prior):
    rows, cols = dark_prior.shape
    flat = dark_prior.flatten()
    flat.sort()
    num = int(rows * cols * 0.001)
    threshold = flat[-num]
    max_pix = img_arr[dark_prior >= threshold]
    max_pix.sort(axis=0)
    return max_pix[-num:, :].mean(axis=0).astype(np.uint8)


if __name__ == '__main__':
    img1 = Image.open("img.png")
    img1_arr = np.asarray(img1)

    D = get_dark_prior(img1_arr)
    A = get_air_light(img1_arr, D)
    T = get_transmission(img1_arr, A)
