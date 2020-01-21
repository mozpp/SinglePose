import numpy.matlib as mt
import numpy as np
import matplotlib.pyplot as plt
import math


def put_heatmap_org(heatmap, plane_idx, center, sigma):
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 1.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    # gaussian filter
    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
            heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)


def put_heatmap(heatmap, plane_idx, center, sigma):
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 1.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    # added
    h = y1 - y0
    w = x1 - x0
    if h <= 0 or w <= 0:
        return

    #  this is right in the center but orginal code is not
    # x_mask = mt.repmat((w - 1)//2, h, w)
    # y_mask = mt.repmat((h - 1)//2, h, w)

    # with one pixel offset
    x_mask = mt.repmat((w + 2)//2, h, w)
    y_mask = mt.repmat((h + 2)//2, h, w)

    x_map = mt.repmat(np.arange(w), h, 1)
    y_map = mt.repmat(np.arange(h), w, 1)
    y_map = np.transpose(y_map)

    d = (x_map - x_mask)**2 + (y_map - y_mask)**2
    in_exp = d / 2. / sigma / sigma
    in_exp = np.where(in_exp > th, np.inf, in_exp)
    gauss = np.exp( -1 * in_exp)

    heatmap[plane_idx][y0:y1, x0:x1] = np.maximum(heatmap[plane_idx][y0:y1, x0:x1], gauss)
    heatmap[plane_idx][y0:y1, x0:x1] = np.minimum(heatmap[plane_idx][y0:y1, x0:x1], 1.0)


def gen_from_matrix(image, center, sigma):
    y, x = center
    h, w = image.shape

    mask_x = mt.repmat(x, h, w)
    mask_y = mt.repmat(y, h, w)

    x1 = np.arange(w)
    x_map = mt.repmat(x1, h, 1)

    y1 = np.arange(h)
    y_map = mt.repmat(y1, w, 1)
    y_map = np.transpose(y_map)

    gauss_map = np.sqrt((x_map - mask_x)**2 + (y_map - mask_y)**2)
    np.exp( -0.5 * gauss_map / sigma)
    return gauss_map


if __name__ == '__main__':
                                                     
    center = [34, 34]
    image = np.zeros([30, 30])
    sigma = 3
    # gauss_map = gen_from_matrix(image, center, sigma)


    heatmap = np.zeros([1, 30, 30])
    heatmap = np.concatenate([heatmap, heatmap], axis=0)

    put_heatmap(heatmap, 0, center, sigma)
    put_heatmap_org(heatmap, 1, center, sigma)

    # print('matrix ==>',heatmap[0])
    # print('orginal ==>',heatmap[1])

    print(np.sum(np.abs(heatmap[0] - heatmap[1]) / (heatmap[1] + 1e-9)))

    fig1 = plt.figure()
    plt.imshow(heatmap[0], plt.cm.gray)

    fig2 = plt.figure()
    plt.imshow(heatmap[1], plt.cm.gray)

    plt.show()
