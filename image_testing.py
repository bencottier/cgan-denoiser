# from scipy.ndimage import rotate as imrotate
import scipy.misc
import numpy as np
from utils import imread
import matplotlib.pyplot as plt
import math


def translate_random(data_list, max_translate):
    if max_translate:
        tx = np.random.randint(-max_translate, max_translate + 1)
        ty = np.random.randint(-max_translate, max_translate + 1)
        data_list = [translate(d, tx, ty) for d in data_list]
    return data_list


def translate(a, tx, ty):
    ax, ay = a.shape
    a_ = np.zeros_like(a)

    def translation_bounds(tx, ty, ax, ay):
        return max(tx, 0), min(ax + tx, ax), max(ty, 0), min(ay + ty, ay)

    x1, x2, y1, y2 = translation_bounds(-tx, -ty, ax, ay)
    x3, x4, y3, y4 = translation_bounds(tx, ty, ax, ay)

    a_[x3:x4, y3:y4] = a[x1:x2, y1:y2]
    return a_


def rotate_random(data_list, max_rotate):
    if max_rotate:
        angle = np.random.uniform(-max_rotate, max_rotate)
        data_list = [rotate(d, angle) for d in data_list]
    return data_list


def rotate_rand90(data_list):
    if np.random.random() > 0.5:
        data_list = [rotate(d, 90) for d in data_list]
    return data_list


def rotate(a, degrees):
    return scipy.misc.imrotate(a, degrees)
    

def main():

    img = imread('/home/ben/projects/honours/nn_artefact_removal/data/train/0.jpg')
    y, x = img[:, :256], img[:, 256:512]
    plt.imshow(img)
    plt.show()

    # for i in range(10):
    #     yt, xt = translate_random([y, x], 10)
    #     # plt.imshow(np.concatenate((yt, xt), axis=1))
    #     plt.imshow(yt)
    #     plt.show()

    for i in range(10):
        yr, xr = rotate_random([y, x], 180)
        # yr, xr = rotate_rand90([y, x])
        plt.imshow(np.concatenate((yr, xr), axis=1))
        # plt.imshow(yr)
        plt.show()


if __name__ == '__main__':
    main()
