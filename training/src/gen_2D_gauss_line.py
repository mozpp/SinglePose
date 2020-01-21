import numpy as np
import numpy.matlib as mt
import cv2
import math
import time, os
import matplotlib.pyplot as plt
from PIL import Image

from gen_connection_hm import gen_hm_of_a_point, get_coords_by_index, show_grayscale, save_grapyscale, gen_hm_of_a_line
from demo import plot_heat
import json
from pycocotools.coco import COCO

def gen_a_gauss_point(hm, center, sigma):
    center_x, center_y = center
    height, width = hm.shape[:2]

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
    x_mask = mt.repmat((w - 1)//2, h, w)
    y_mask = mt.repmat((h - 1)//2, h, w)

    # # with one pixel offset
    # x_mask = mt.repmat((w + 2)//2, h, w)
    # y_mask = mt.repmat((h + 2)//2, h, w)

    x_map = mt.repmat(np.arange(w), h, 1)
    y_map = mt.repmat(np.arange(h), w, 1)
    y_map = np.transpose(y_map)

    d = (x_map - x_mask)**2 + (y_map - y_mask)**2
    in_exp = d / 2. / sigma / sigma
    in_exp = np.where(in_exp > th, np.inf, in_exp)
    gauss = np.exp( -1 * in_exp)

    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], gauss)
    hm[y0:y1, x0:x1] = np.minimum(hm[y0:y1, x0:x1], 1.0)
    return hm

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def put_heatmap_line(heatmap, plane_idx, p1, p2, sigma):

    x_p1, y_p1 = p1
    x_p2, y_p2 = p2

    _, height, width = heatmap.shape[:3]

    th = 1.6052
    delta = math.sqrt(th * 2)

    x_p3 = int(max(0, min(x_p1, x_p2) - delta * sigma))
    y_p3 = int(max(0, min(y_p1, y_p2) - delta * sigma))
    p3 = (x_p3, y_p3)

    x_p4 = int(min(width, max(x_p1, x_p2) + delta * sigma))
    y_p4 = int(min(height, max(y_p1, y_p2) + delta * sigma))
    p4 = (x_p4, y_p4)


    E = int(distance(p3, p4))


    x_c, y_c = (x_p1 + x_p2) / 2, (y_p1 + y_p2) / 2
    center = (x_c, y_c)

    hm_crop = np.zeros([E, E])
    hm_crop = gen_a_gauss_point(hm_crop, (E//2+1, E//2+1), sigma)
    # plt.imshow(hm_crop, plt.cm.gray)
    # plt.show()

    L = int(distance(p1, p2))
    if L <= 0:
        return

    hm_crop = stretch(hm_crop, (E//2, E//2), L)
    # plt.imshow(hm_crop, plt.cm.gray)
    # plt.show()


    hm_crop = rotate(hm_crop, p1, p2)
    # plt.imshow(hm_crop, plt.cm.gray)
    # plt.show()

    hm_crop = hm_crop[int(L/2) : int(L/2) + E]
    # plt.imshow(hm_crop, plt.cm.gray)
    # plt.show()

    x_p5 = x_c - E//2
    y_p5 = y_c - E//2
    x_p6 = x_c + E//2
    y_p6 = y_c + E//2

    x_p7 = int(max(0, x_p5))
    y_p7 = int(max(0, y_p5))
    x_p8 = int(min(width,  x_p6))
    y_p8 = int(min(height, y_p6))

    x_p9 = int(0 if x_p5 >= 0 else -1 * x_p5)
    y_p9 = int(0 if y_p5 >= 0 else -1 * y_p5)
    x_p10 = int(E if x_p6 <= width  else E - (x_p6 - width))
    y_p10 = int(E if y_p6 <= height else E - (y_p6 - height))

    dx = x_p8 - x_p7
    dy = y_p8 - y_p7

    if dx <= 0 or dy <= 0 or y_p10 - y_p9 <= 0 or x_p10 - x_p9 <= 0:
        return
    patch = np.array(Image.fromarray(hm_crop[y_p9:y_p10, x_p9:x_p10]).resize([dx, dy]))
    heatmap[plane_idx][y_p7:y_p8, x_p7:x_p8] = np.maximum(heatmap[plane_idx][y_p7:y_p8, x_p7:x_p8],
                                                          patch)
    heatmap[plane_idx][y_p7:y_p8, x_p7:x_p8] = np.minimum(heatmap[plane_idx][y_p7:y_p8, x_p7:x_p8], 1.0)

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
    x_mask = mt.repmat((w - 1)//2, h, w)
    y_mask = mt.repmat((h - 1)//2, h, w)

    # # with one pixel offset
    # x_mask = mt.repmat((w + 2)//2, h, w)
    # y_mask = mt.repmat((h + 2)//2, h, w)

    x_map = mt.repmat(np.arange(w), h, 1)
    y_map = mt.repmat(np.arange(h), w, 1)
    y_map = np.transpose(y_map)

    d = (x_map - x_mask)**2 + (y_map - y_mask)**2
    in_exp = d / 2. / sigma / sigma
    in_exp = np.where(in_exp > th, np.inf, in_exp)
    gauss = np.exp( -1 * in_exp)

    heatmap[plane_idx][y0:y1, x0:x1] = np.maximum(heatmap[plane_idx][y0:y1, x0:x1], gauss)
    heatmap[plane_idx][y0:y1, x0:x1] = np.minimum(heatmap[plane_idx][y0:y1, x0:x1], 1.0)

def stretch(img, center, length):
    assert int(length) > 0
    center_x, center_y = center
    center_line = img[center_y]
    center_plane = np.stack([center_line] * int(length), axis=0)
    leng_img = np.concatenate([img[:center_y], center_plane, img[center_y:]])

    return leng_img

def rotate(img, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    if y1 == y2:
        theta = 90
    else:
        theta = np.arctan((x2-x1)/(y2-y1)) * 180 / np.pi
    return np.array(Image.fromarray(img).rotate(theta))


if __name__ == '__main__':
    ''' test hm using plt '''
    # while True:
    #     sigma = 3
    #     heatmap = np.zeros([1, 50, 50])
    #     heatmap = np.concatenate([heatmap, heatmap], axis=0)
    #
    #     # put_heatmap(heatmap, 0, center, sigma)
    #     # img = stretch(heatmap[0], center, 20)
    #     # img = rotate(img, (0,0),(1,1))
    #
    #     # p1=(20,20)
    #     # p2=(31,20)
    #
    #     # p1, p2 = [35,  7], [32, 26]
    #
    #     p1 = np.random.randint(0, 50, [2])
    #     p2 = np.random.randint(0, 50, [2])
    #     print(p1,p2)
    #
    #     put_heatmap_line(heatmap, 0, p1, p2, sigma)
    #
    #     # img = gen_a_gauss_point(np.zeros([50, 40]), center, sigma)
    #     fig1 = plt.figure()
    #     plt.scatter(p1[0], p1[1], c='r', linewidths=2)
    #     plt.scatter(p2[0], p2[1], c='b', linewidths=2)
    #
    #     plt.imshow(heatmap[0], plt.cm.gray)
    #     # plt.imshow(img, plt.cm.gray)
    #
    #
    #
    #     plt.show()




    ''' test hm using cv2, sometimes got stucked'''
    # while True:
    #     sigma = 3
    #     heatmap = np.zeros([1, 500, 500])
    #     heatmap = np.concatenate([heatmap, heatmap], axis=0)
    #
    #     p1 = tuple(np.random.randint(0, 500, [2]))
    #     p2 = tuple(np.random.randint(0, 500, [2]))
    #     print(p1, p2)
    #
    #     put_heatmap_line(heatmap, 0, p1, p2, sigma)
    #     cv2.line(heatmap[0], p1,p1,color=(0,0,255), thickness=2)
    #     cv2.line(heatmap[0], p2,p2,color=(255,0,0), thickness=2)
    #
    #     cv2.imshow('test', heatmap[0])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    ''' test entire hms '''
    # test img from dataset
    dataset_json_path = "../dataset/aich_plus_coco/ai_challenger_train.json"
    dataset_path = "../dataset"

    anno = COCO(dataset_json_path)
    imgIds = anno.getImgIds()
    imgId = imgIds[2]

    img_meta = anno.loadImgs([imgId])[0]

    anno_ids = anno.getAnnIds(imgIds=imgId)
    img_anno = anno.loadAnns(anno_ids)

    img_path = os.path.join(dataset_path, img_meta['file_name'])

    ann = img_anno[0]

    # test costumized img

    # ann = {"num_keypoints": 13,
    #        "area": 517771,
    #        "keypoints": [287, 84, 2, 291, 207, 1, 388, 213, 2, 205, 279, 2, 523, 273, 2, 171, 382, 2, 592, 272, 2, 91,
    #                      474, 2, 395, 479, 2, 298, 483, 2, 384, 648, 2, 235, 648, 2, 319, 863, 2, 154, 805, 2],
    #        "bbox": [44, 61, 607, 853],
    #        "image_id": 638,
    #        "category_id": 1,
    #        "id": 638}
    # img_path = "/media/yangfeiyu/560dccd6-4f69-40e3-9d5b-c5d93d907f80/yangfeiyu/DataSets/" + \
    #            "ai_challenger/valid/1a98fcb21be0c72e0919fa98b1e3c8edd08cb70e.jpg"


    # points, joints
    parts = ["top_head", "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
             "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    num_points = len(parts)
    # first level limbs, straight line
    first_level_limbs = [(0, 1), (1, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7), (1, 8), (8, 10), (10, 12), (1, 9),
                         (9, 11), (11, 13)]
    num_fst_limbs = len(first_level_limbs)
    # second level limbs, broken line
    second_level_limbs = [(2, 3), (5, 6), (8, 9), (11, 12), (0, 1, 4, 7, 10)]
    num_sec_limbs = len(second_level_limbs)
    # third level limbs, entire body
    third_level_limbs = list(range(num_fst_limbs))

    hm_thickness = 2

    show = True
    show_hm = False
    if show:
        # show image
        img = cv2.imread(img_path)
        for index in range(num_points):
            p = tuple(ann["keypoints"][3 * index: 3 * index + 2])
            cv2.line(img, p, p, color=(255, 0, 0), thickness=10)
            cv2.putText(img, parts[index], p, fontFace=cv2.FONT_ITALIC, fontScale=0.5, color=(0, 0, 255), thickness=1)
        cv2.imshow('test', img)
        cv2.imwrite('../demo_io/limb_hm/img.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    heatmap = np.zeros([num_points + num_fst_limbs + num_sec_limbs + 1, 1000, 666])

    # build point hm
    for index in range(num_points):
        gen_hm_of_a_point(heatmap, index, get_coords_by_index(ann, index), hm_thickness)
        if show_hm:
            show_grayscale(heatmap[index])

    # build 1st limb hm
    for index, limb in enumerate(first_level_limbs):
        start = get_coords_by_index(ann, limb[0])
        end = get_coords_by_index(ann, limb[1])

        hm_index = num_points + index
        # new way of gauss line generating
        old = False
        if old:
            gen_hm_of_a_line(heatmap, hm_index, start, end, sparsity=20, sigma=hm_thickness)
        else:
            put_heatmap_line(heatmap, hm_index, start, end, sigma=hm_thickness)

        if show_hm:
            show_grayscale(heatmap[hm_index])

    # build 2nd limb hm, assemble form 1st level
    for index, limb in enumerate(second_level_limbs):
        hm_index = num_points + num_fst_limbs + index

        sec_limb_indice = num_points + np.array(limb)
        heatmap[hm_index] = np.max(heatmap[sec_limb_indice, :, :], axis=0)
        if show_hm:
            show_grayscale(heatmap[hm_index])

    # build 3rd limb hm, assemble form 1st level
    sec_limb_indice = num_points + np.array(third_level_limbs)
    heatmap[-1] = np.max(heatmap[sec_limb_indice, :, :], axis=0)
    if show:
        show_grayscale(heatmap[-1])

    # show or save hm
    for index in range(heatmap.shape[0]):
        # show_grayscale(heatmap[index])
        save_grapyscale(heatmap[index], '../demo_io/limb_hm/' + str(index) + '.png')


    plot_heat(img, np.transpose(heatmap, [1,2,0]))

