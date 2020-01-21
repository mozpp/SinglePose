from __future__ import print_function
from __future__ import division

import cv2
import numpy as np

def get_cross_point(l1, l2, shape):
    k1 = 1.0 * (l1[1] - l1[3]) / (l1[0] - l1[2])
    k2 = 1.0 * (l2[1] - l2[3]) / (l2[0] - l2[2])
    x = (k1 * l1[0] - l1[1] - k2 * l2[0] + l2[1]) / (k1 - k2)
    y = (k1*k2*(l1[0]-l2[0]) + k1*l2[1] - k2*l1[1]) / (k1 - k2)

    if 0 < x < shape[1] and 0 < y < shape[0]:
        return (int(x), int(y))
    else:
        return None

def unit_vector(line):
    x1, y1, x2, y2 = line
    d = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return ((x2 - x1) / d, (y2 - y1) / d)

def inter_product(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    return x1*x2 + y1*y2

def get_cross_point_from_heatmap(heatmap_, theta):
    # print(type(heatmap_.reshape([-1])[0]), heatmap_.shape)
    heatmap = heatmap_.copy()
    heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)

    if heatmap.shape[-1] == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    heatmap = cv2.threshold(heatmap, 100, 255, cv2.THRESH_BINARY)[1]

    heatmap = cv2.Canny(heatmap, 10, 20)

    # cv2.imshow('lines', heatmap)
    # cv2.waitKey(0)

    # heatmap = cv2.erode(heatmap, cv2.MORPH_RECT, iterations=2)
    # heatmap = cv2.erode(heatmap, None, iterations=1)
    # thresh = cv2.dilate(thresh, None, iterations=2)


    lines = cv2.HoughLinesP(heatmap, rho=1, theta=np.pi/180, threshold=10, lines=50,
                            minLineLength=1, maxLineGap=100)

    if lines is None:
        print("no line detected")
        return None

    # for l in lines:
    #     l = l[0]
    #     cv2.line(heatmap, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2, 8)
    # cv2.imshow('lines', heatmap)
    # cv2.waitKey(0)

    l1 = lines[0][0]
    l2 = None

    l1_unit_vector = unit_vector(l1)
    for l in lines[1:]:
        l = l[0]
        l_unit_vector = unit_vector(l)
        if np.abs(inter_product(l1_unit_vector, l_unit_vector)) < np.cos(theta * np.pi / 180):
            l2 = l
            break

    if l2 is None:
        print("cannot find l2")
        return None

    # print(l1, l2)
    # cv2.line(heatmap, (l1[0], l1[1]), (l1[2], l1[3]), (0, 0, 255), 2, 8)
    # cv2.line(heatmap, (l2[0], l2[1]), (l2[2], l2[3]), (0, 0, 255), 2, 8)
    # cv2.imshow('heatmap', heatmap)
    # cv2.waitKey(0)

    cross_point = get_cross_point(l1, l2, heatmap.shape)
    if cross_point is None:
        print("Wrong cross point")
        return None

    target_line = (l1, l2)
    return cross_point, target_line, lines


if __name__ == '__main__':
    for i in range(27, 31):
        # path = '../../demo_io/limb_hm/{:d}.png'.format(i)
        path = '../../demo_io/img_and_heat_conn/{:d}.png'.format(i)

        print(path)


        heatmap = cv2.imread(path)
        ret = get_cross_point_from_heatmap(heatmap, theta=30)

        if not ret is None:
            cross_point, target_line, lines = ret
            axs[i].scatter(cross_point[0], cross_point[1], s=10, c="r")

            l1, l2 = target_line

            # for l in lines:
            #     l = l[0]
            #     cv2.line(heatmap, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2, 8)
            # cv2.imshow('lines', heatmap)
            # cv2.waitKey(0)

            cv2.line(heatmap, (l1[0], l1[1]), (l1[2], l1[3]), (0, 255, 0), 2, 8)
            cv2.line(heatmap, (l2[0], l2[1]), (l2[2], l2[3]), (255, 0, 0), 2, 8)
            cv2.circle(heatmap, cross_point, 6, (0, 0, 255))

            cv2.imshow('heatmap', heatmap)
            cv2.waitKey(0)
        else:
            from end_points_detect import get_end_points_with_pca
            ret = get_end_points_with_pca(heatmap)
            if not ret is None:
                p1, p2, cntr, contour = ret
                cv2.circle(heatmap, cntr, 6, (0, 0, 255))
                cv2.imshow('heatmap', heatmap)
                cv2.waitKey(0)
            else:
                continue