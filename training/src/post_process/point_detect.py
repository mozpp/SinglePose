from __future__ import print_function
from __future__ import division
import cv2 as cv
import cv2
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi
from scipy.ndimage.filters import gaussian_filter



def get_point(heatmap_):
    heatmap = heatmap_.copy()
    # heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2GRAY)
    # heatmap = gaussian_filter(heatmap, sigma=5)

    p = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (int(p[1]), int(p[0]))


if __name__ == '__main__':
    for i in range(0, 14):
    # for i in range(0, 33):
        # path = '../../demo_io/limb_hm/{:d}.png'.format(i)
        path = '../../demo_io/img_and_heat_conn/{:d}.png'.format(i)
        print(path)

        heatmap = cv.imread(path)

        # Check if image is loaded successfully
        if heatmap is None:
            print('Could not open or find the image: ', args.input)
            exit(0)

        ret = get_point(heatmap)

        if not ret is None:
            point = ret
            cv.circle(heatmap, point, 3, (255, 0, 255), 2)
            cv.imshow('output', heatmap)
            cv.waitKey()
        else:
            continue
