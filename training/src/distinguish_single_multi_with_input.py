# -*- coding: utf-8 -*-
# @Time    : 18-7-10 上午9:41
# @Author  : zengzihua@huya.com
# @FileName: benchmark.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import json
import argparse
import cv2
import os, shutil
import math
import time

from scipy.ndimage.filters import gaussian_filter
from pPose_nms import write_json
from fn import vis_frame
from scipy.ndimage.filters import gaussian_filter


def get_hm_extremums(heatmap, gauss_sigma):
    # filt
    heatmap = gaussian_filter(heatmap, sigma=gauss_sigma)
    # padding
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 1:heatmap_with_borders.shape[1] - 1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 0:heatmap_with_borders.shape[1] - 2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1] - 1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0] - 2, 1:heatmap_with_borders.shape[1] - 1]
    # peak mask with padding
    heatmap_peaks = (heatmap_center > heatmap_left) & \
                    (heatmap_center > heatmap_right) & \
                    (heatmap_center > heatmap_up) & \
                    (heatmap_center > heatmap_down)
    # peak mask without padding
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0] - 1, 1:heatmap_center.shape[1] - 1]
    return heatmap_peaks

def pose_processing(hm, coord):
    px = int(math.floor(coord[0] + 0.5))
    py = int(math.floor(coord[1] + 0.5))
    if 1 < px < hm.shape[0] - 1 and 1 < py < hm.shape[1] - 1:
        diff = np.array([hm[py][px + 1] - hm[py][px - 1], hm[py + 1][px] - hm[py - 1][px]])
        coord += np.sign(diff) * 0.25
    coord = np.array(coord).astype(np.float32)
    coord += 0.2

    return coord


def cal_coord(pred_heatmaps, images_anno, args):
    final_result = []
    for img_id in pred_heatmaps.keys():
        heat_h, heat_w, n_kpoints = pred_heatmaps[img_id].shape
        scale_h, scale_w = heat_h / images_anno[img_id]['height'], heat_w / images_anno[img_id]['width']
        coord = []
        scores = []
        result = []
        for p_ind in range(n_kpoints):
            heat = pred_heatmaps[img_id][:, :, p_ind]
            scores.append(np.max(heat))
            heat = gaussian_filter(heat, sigma=5)
            ind = np.unravel_index(np.argmax(heat), heat.shape)
            ind = pose_processing(heat, ind)
            coord_x = int((ind[1] + 1) / scale_w)
            coord_y = int((ind[0] + 1) / scale_h)
            coord.append((coord_x, coord_y))
        result.append({
            'keypoints': coord,
            'kp_score': scores})
        result = {
            'imgname': images_anno[img_id]['file_name'],
            'result': result
        }
        if args.save_imgs:
            ori_img = cv2.imread(os.path.join(args.img_path, images_anno[img_id]['file_name']))
            # ori_img = cv2.resize(ori_img, (480, 640))
            img = vis_frame(ori_img, result)
            cv2.imwrite(os.path.join(args.outputpath, images_anno[img_id]['file_name']), img)
        final_result.append(result)
    return final_result


def infer(frozen_pb_path, output_node_name, img_path, images_anno):
    with tf.gfile.GFile(frozen_pb_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name("image:0")
    output_heat = graph.get_tensor_by_name("%s:0" % output_node_name)

    res = {}
    use_times = []
    with tf.Session() as sess:
        for img_id in images_anno.keys():
            ori_img = cv2.imread(os.path.join(img_path, images_anno[img_id]['file_name']))
            shape = input_image.get_shape().as_list()
            inp_img = cv2.resize(ori_img, (shape[1], shape[2]))
            st = time.time()
            heat = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            infer_time = 1000 * (time.time() - st)
            print("img_id = %d, cost_time = %.2f ms" % (img_id, infer_time))
            use_times.append(infer_time)
            res[img_id] = np.squeeze(heat)
    print("Average inference time = %.2f ms" % np.mean(use_times))
    return res


def pre():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)

def calc_max_and_submax(joint_specified_hm, init_gauss_sigma=10):
    # calc mask and ensure we get at least two maximums
    hm_ext_mask = np.zeros_like(joint_specified_hm)
    gauss_sigma = init_gauss_sigma

    hm_ext_mask = get_hm_extremums(joint_specified_hm, gauss_sigma=gauss_sigma)

    while np.sum(hm_ext_mask) < 2:
        gauss_sigma = gauss_sigma // 2
        hm_ext_mask = get_hm_extremums(joint_specified_hm, gauss_sigma=gauss_sigma)

        # no extremum
        if gauss_sigma == 0 and np.sum(hm_ext_mask) == 0:
            max_val, sub_max_val = 0, 0
            return max_val, sub_max_val

        # one extremum
        if gauss_sigma == 0 and np.sum(hm_ext_mask) == 1:
            max_val, sub_max_val = int(np.max(joint_specified_hm)), 0
            return max_val, sub_max_val

    if np.sum(hm_ext_mask) >= 2:
        hm_ext_vals = np.squeeze(joint_specified_hm[hm_ext_mask])
        hm_ext_vals.sort()
        sub_max_val, max_val = hm_ext_vals[-2:]
        return max_val, sub_max_val


def eval_accuracy(mean_diff, images_anno, threshold, verbose=False):
    num_imgs = len(mean_diff)
    pred_is_one_person = mean_diff > threshold
    gt_is_one_person = np.array([images_anno[img_id]['is_one_person'] for img_id in range(num_imgs)])
    accuracy = np.sum(gt_is_one_person == pred_is_one_person) / num_imgs

    print('Threshold: %d, accuracy: %.2f' % (threshold, accuracy))
    if verbose:
        for img_id in range(num_imgs):
            print(images_anno[img_id]['file_name'], pred_is_one_person[img_id])

if __name__ == '__main__':
    pre()

    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--frozen_pb_path", type=str,
                        # default="../mv2_cpm_tiny/models/mv2_chk/mobilenetv2_model_592000_224x224_tf1.14.pb",
                        default="../mv2_cpm_tiny/models/mv2_chk/mv2_chk-320x320.pb",
                        )
    parser.add_argument("--img_path", type=str, default="../demo_io/inputs")
    parser.add_argument("--output_node_name", type=str, default="Convolutional_Pose_Machine/stage_5_out")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--save_imgs", type=bool, default=True)
    parser.add_argument("--outputpath", type=str, default="../demo_io/outputs")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # anno = json.load(open(args.anno_json_path))

    images_anno = {}
    filelist = os.listdir(args.img_path)
    # filelist.sort(key=lambda x: int(x[:-4]))
    print("Total test example=%d" % len(filelist))
    for index, item in enumerate(filelist):
        img = cv2.imread(os.path.join(args.img_path, filelist[index]))
        img_info = {}
        img_info['file_name'] = item
        img_info['height'] = img.shape[0]
        img_info['width'] = img.shape[1]
        img_info['id'] = index
        img_info['is_one_person'] = False if len(item) == 16 else True
        images_anno[index] = img_info

    pred_heatmap = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)

    # normalize hm
    normalized_hm = pred_heatmap.copy()
    for k, v in normalized_hm.items():
        max_val = np.max(v)
        normalized_hm[k] = 255. * v / max_val

    num_imgs = len(normalized_hm.keys())
    num_kps = normalized_hm[1].shape[-1]
    diff = np.ones([num_imgs, num_kps]) * -1

    for joint in range(num_kps):
        # create output folders
        if args.save_imgs:
            joint_file_name = os.path.join(args.outputpath, str(joint))
            if os.path.exists(joint_file_name):
                shutil.rmtree(joint_file_name)
            os.makedirs(joint_file_name)

        # calc max and submax
        for img_id, hm in normalized_hm.items():
            joint_specified_hm = hm[:, :, joint]
            max_val, sub_max_val = calc_max_and_submax(joint_specified_hm, init_gauss_sigma=10)
            diff[img_id, joint] = max_val - sub_max_val

            # save hm
            if args.save_imgs:
                img_file_name = images_anno[img_id]['file_name']
                index = img_file_name.find('.')
                img_file_name = img_file_name[:index] + '_' + str(int(max_val)) + '_' + str(int(sub_max_val)) + img_file_name[index:]
                joint_hm_name = os.path.join(joint_file_name, img_file_name)
                cv2.imwrite(joint_hm_name, joint_specified_hm)


    mean_diff = np.mean(diff, axis=1)
    thresholds = np.arange(50,150,10)
    for th in thresholds:
        eval_accuracy(mean_diff, images_anno, th, verbose=False)

    final_result = cal_coord(pred_heatmap, images_anno, args)

    write_json(final_result, args.outputpath)