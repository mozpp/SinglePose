from __future__ import print_function
from __future__ import division
import cv2 as cv
import cv2
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(contour):
    sz = len(contour)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = contour[i, 0, 0]
        data_pts[i, 1] = contour[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    projects = cv2.PCAProject(data_pts, mean, np.array([eigenvectors[0]]))

    id_min = np.argmin(projects)
    id_max = np.argmax(projects)

    pt_min = data_pts[id_min]
    pt_max = data_pts[id_max]

    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    p1 = (int(pt_min[0]), int(pt_min[1]))
    p2 = (int(pt_max[0]), int(pt_max[1]))

    return p1, p2, cntr


def get_end_points_with_pca(heatmap_):
    # print(type(heatmap_.reshape([-1])[0]), heatmap_.shape)
    heatmap = heatmap_.copy()
    heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    if heatmap.shape[-1] == 3:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2GRAY)

    _, bw = cv.threshold(heatmap, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # bw = cv2.erode(bw, None, iterations=2)
    # bw = cv2.dilate(bw, None, iterations=2)

    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    contour = None
    tmp_area = -1
    for i, c in enumerate(contours):
        if cv.contourArea(c) > tmp_area:
            contour = c

    if contour is None:
        print("detect no contour")
        return None

    p1, p2, cntr = getOrientation(contour)

    return p1, p2, cntr, contour


if __name__ == '__main__':
    for i in range(14, 26):
    # for i in range(0, 33):
        # path = '../../demo_io/limb_hm/{:d}.png'.format(i)
        path = '../../demo_io/img_and_heat_conn/{:d}.png'.format(i)
        print(path)

        heatmap = cv.imread(path)

        # Check if image is loaded successfully
        if heatmap is None:
            print('Could not open or find the image: ', args.input)
            exit(0)

        ret = get_end_points_with_pac(heatmap)

        if not ret is None:
            p1, p2, cntr, contour = ret
            cv.circle(heatmap, p1, 3, (255, 0, 255), 2)
            cv.circle(heatmap, p2, 3, (255, 0, 255), 2)
            cv.drawContours(heatmap, [contour], 0, (0, 0, 255), 1)
            cv.imshow('output', heatmap)
            cv.waitKey()
        else:
            continue
