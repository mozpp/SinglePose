import numpy as np
import numpy.matlib as mt
import cv2
import math
import time

def put_heatmap_conn(heatmap, plane_idx, center, sigma):
    old = False
    if old:
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

    else:
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 1.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        # print('delat*sigma, center_y, y1, y0,h==>', delta*sigma, center_y, y1, y0, y1 - y0)
        # added
        h = y1 - y0
        w = x1 - x0

        if h <= 0 or w <= 0:
            return

        #  this is right in the center but orginal code is not
        # x_mask = mt.repmat((w - 1)//2, h, w)
        # y_mask = mt.repmat((h - 1)//2, h, w)

        # with one pixel offset
        x_mask = mt.repmat((w + 2) // 2, h, w)
        y_mask = mt.repmat((h + 2) // 2, h, w)

        x_map = mt.repmat(np.arange(w), h, 1)
        y_map = mt.repmat(np.arange(h), w, 1)
        y_map = np.transpose(y_map)

        d = (x_map - x_mask) ** 2 + (y_map - y_mask) ** 2
        in_exp = d / 2. / sigma / sigma
        in_exp = np.where(in_exp > th, np.inf, in_exp)
        gauss = np.exp(-1 * in_exp)

        heatmap[plane_idx][y0:y1, x0:x1] = np.maximum(heatmap[plane_idx][y0:y1, x0:x1], gauss)
        heatmap[plane_idx][y0:y1, x0:x1] = np.minimum(heatmap[plane_idx][y0:y1, x0:x1], 1.0)


def gen_hm_of_a_point(heatmap, plane_idx, center, sigma=2):
    return put_heatmap_conn(heatmap, plane_idx, center, sigma)

def gen_hm_of_multi_points(heatmap, plane_idx, centers, sigma=2):
    _, h, w = heatmap.shape[:3]
    hm_slice_before_fuse = np.zeros([len(centers), h, w])
    for ind, center in enumerate(centers):
        gen_hm_of_a_point(hm_slice_before_fuse, ind, center, sigma)
    hm_slice_after_fuse = np.max(hm_slice_before_fuse, axis=0)

    heatmap[plane_idx] = hm_slice_after_fuse

def normlize_grayscale_image(grayscale_img):
    x = grayscale_img
    return (255 * x / np.max(x)).astype(np.uint8)

def show_grayscale(grayscale_img):
    cv2.imshow('show', normlize_grayscale_image(grayscale_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_grapyscale(grayscale_img, path):
    img = normlize_grayscale_image(grayscale_img)
    cv2.imwrite(path, img)

def gen_points_along_a_line(start, end, num_points=1000):
    return np.linspace(start, end, num_points)

def get_coords_by_index(ann, index):
    return tuple(ann["keypoints"][3 * index: 3 * index + 2])

def gen_hm_of_a_line(heatmap, plane_idx, start, end, sparsity, sigma=2):
    points = gen_points_along_a_line(start, end)
    points = np.unique(points.astype(np.int32), axis=0)

    # interpolate
    points = points[::sparsity,:]

    # print('points ==>', len(points))
    gen_hm_of_multi_points(heatmap, plane_idx, points, sigma)



if __name__ == '__main__':



    ann = {"num_keypoints": 13,
           "area": 517771,
           "keypoints": [287, 84, 2, 291, 207, 1, 388, 213, 2, 205, 279, 2, 523, 273, 2, 171, 382, 2, 592, 272, 2, 91, 474, 2, 395, 479, 2, 298, 483, 2, 384, 648, 2, 235, 648, 2, 319, 863, 2, 154, 805, 2],
           "bbox": [44, 61, 607, 853],
           "image_id": 638,
           "category_id": 1,
           "id": 638}
    img_path = "/media/yangfeiyu/560dccd6-4f69-40e3-9d5b-c5d93d907f80/yangfeiyu/DataSets/" + \
               "ai_challenger/valid/1a98fcb21be0c72e0919fa98b1e3c8edd08cb70e.jpg"

    # points, joints
    parts = ["top_head", "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    num_points = len(parts)
    # first level limbs, straight line
    first_level_limbs = [(0,1), (1,2), (2,4),(4,6),(1,3),(3,5),(5,7),(1,8),(8,10),(10,12),(1,9), (9,11),(11,13)]
    num_fst_limbs = len(first_level_limbs)
    # second level limbs, broken line
    second_level_limbs = [(2, 3), (5,6), (8,9),(11,12), (0,1,4,7,10)]
    num_sec_limbs = len(second_level_limbs)
    # third level limbs, entire body
    third_level_limbs = list(range(num_fst_limbs))

    hm_thickness = 1



    show = True
    show_hm = False
    if show:
        # show image with kps
        img = cv2.imread(img_path)
        for index in range(num_points):
            p = tuple(ann["keypoints"][3 * index: 3 * index + 2])
            cv2.line(img, p, p, color=(255, 0, 0), thickness=10)
            cv2.putText(img, parts[index], p, fontFace=cv2.FONT_ITALIC, fontScale=0.5,color=(0, 0, 255), thickness=1)
        cv2.imshow('test', img)
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
        gen_hm_of_a_line(heatmap, hm_index, start, end, sparsity=4, sigma=hm_thickness)
        if show_hm:
            show_grayscale(heatmap[hm_index])

    # build 2nd limb hm, assemble form 1st level
    for index, limb in enumerate(second_level_limbs):
        hm_index = num_points + num_fst_limbs + index

        sec_limb_indice = num_points + np.array(limb)
        heatmap[hm_index] = np.max(heatmap[sec_limb_indice,:,:], axis=0)
        if show_hm:
            show_grayscale(heatmap[hm_index])

    # build 3rd limb hm, assemble form 1st level
    sec_limb_indice = num_points + np.array(third_level_limbs)
    heatmap[-1] = np.max(heatmap[sec_limb_indice,:,:], axis=0)
    if show:
        show_grayscale(heatmap[-1])


    # show or save hm
    for index in range(heatmap.shape[0]):
        # show_grayscale(heatmap[index])
        save_grapyscale(heatmap[index], '../demo_io/limb_hm/' + str(index)+'.png')






