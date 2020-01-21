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
import os
import math
import time

from scipy.ndimage.filters import gaussian_filter


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def calibrate_coord(coord_x, coord_y, heat, delta=13):
    h, w = heat.shape

    x_min = coord_x - delta
    x_max = coord_x + delta
    y_min = coord_y - delta
    y_max = coord_y + delta

    x_min = x_min if x_min > 0 else 0
    x_max = x_max if x_max < w else w
    y_min = y_min if y_min > 0 else 0
    y_max = y_max if y_max < h else h

    xx = np.array([range(x_min, x_max)])
    xx = np.stack([xx] * (y_max - y_min), axis=1)
    yy = np.array([range(y_min, y_max)])
    yy = np.stack([yy] * (x_max - x_min), axis=2)

    heat = heat[y_min:y_max, x_min:x_max]
    score = np.sum(heat)
    xx = np.sum(xx * heat)
    yy = np.sum(yy * heat)

    if score != 0:
        coord_x = xx / score
        coord_y = yy / score
        return coord_x, coord_y
    else:
        return coord_x, coord_y


def cal_coord(pred_heatmaps, images_anno):
    coords = {}
    for img_id in pred_heatmaps.keys():
        heat_h, heat_w, n_kpoints = pred_heatmaps[img_id].shape
        scale_h, scale_w = heat_h / images_anno[img_id]['height'], heat_w / images_anno[img_id]['width']
        coord = []
        for p_ind in range(n_kpoints):
            heat = pred_heatmaps[img_id][:, :, p_ind]

            ### add threshold
            s = np.max(heat)
            if s < 0.05:
                coord.append((0, 0))
                continue
            ###

            heat = gaussian_filter(heat, sigma=8) # 5
            ind = np.unravel_index(np.argmax(heat), heat.shape)

            # coord_x = int((ind[1] + 1) / scale_w)
            # coord_y = int((ind[0] + 1) / scale_h)
            coord_x = ind[1] + 1
            coord_y = ind[0] + 1

            # print('befor calibrate ==> ', coord_x, coord_y)
            coord_x, coord_y = calibrate_coord(coord_x, coord_y, heat)
            # print('after calibrate ==> ', coord_x, coord_y)

            coord_x = int(coord_x / scale_w)
            coord_y = int(coord_y / scale_h)

            coord.append((coord_x, coord_y))
        coords[img_id] = coord
    return coords



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
        count = 0
        for img_id in list(images_anno.keys())[:]:

            ''' add '''
            img_name = images_anno[img_id]['file_name']
            if len(img_name.split('/')) != 3:
                images_anno[img_id]['file_name'] = os.path.join('ai_challenger',
                                                                'train',
                                                                images_anno[img_id]['file_name'])
            ''' end '''
            ori_img = cv2.imread(os.path.join(img_path, images_anno[img_id]['file_name']))


            shape = input_image.get_shape().as_list()
            inp_img = cv2.resize(ori_img, (shape[1], shape[2]))
            # norm
            inp_img = inp_img #/ 255.
            st = time.time()
            heat = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            infer_time = 1000 * (time.time() - st)
            print("img_id = %d, cost_time = %.2f ms" % (img_id, infer_time))
            if count != 0:
                use_times.append(infer_time)
            count = count + 1
            res[img_id] = np.squeeze(heat)
    print("Average inference time = %.2f ms" % np.mean(use_times))
    return res

def pre():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)

if __name__ == '__main__':
    #pre()

    parser = argparse.ArgumentParser(description="PCKh benchmark")
    parser.add_argument("--frozen_pb_path", type=str,
                        # default="../mv2_cpm_tiny/models/mv2_chk/mv2_chk-224x224.pb",
                        # default="../mv2_cpm_tiny/models/mv2_chk/mv2_chk-320x320.pb",
                        # default="../mv2_cpm_tiny/models/mv2_cpm_batch-14_lr-0.001_gpus-1_224x224_..-experiments-mv2_cpm/mv2_cpm_batch-14_lr-0.001_gpus-1_224x224_..-experiments-mv2_cpm.pb", # channel ratio = 1
                        # default="../mv2_cpm_tiny/models/mv2_chk/mobilenetv2_model_592000_224x224_tf1.14.pb",

                        # PCKh=82.43 14pks 224
                          default="/data/home/chenyu/Yangfeiyu/singlePose/log/zq_224/model/l2_symm_94.62/model-304000.pb"
                        # default="../log/zq21_model_70000_224x224_2stage_tf1.14.pb",

                        )
    parser.add_argument("--anno_json_path", type=str,
                          default="../dataset/aich_plus_coco/ai_challenger_valid.json",
                        # default="../dataset/aich_plus_coco/coco_train.json"
                        )
    parser.add_argument("--img_path", type=str,
                        default="../dataset",
                        )
    #parser.add_argument("--output_node_name", type=str, default="Convolutional_Pose_Machine/stage_5_out")
    parser.add_argument("--output_node_name", type=str, default="CPM/stage_2_out")

    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    anno = json.load(open(args.anno_json_path))
    print("Total test example=%d" % len(anno['images']))

    images_anno = {}
    keypoint_annos = {}
    transform = list(zip(
        [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13],
        [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13]
    ))
    for img_info, anno_info in zip(anno['images'], anno['annotations']):
        images_anno[img_info['id']] = img_info

        prev_xs = anno_info['keypoints'][0::3]
        prev_ys = anno_info['keypoints'][1::3]

        new_kp = []
        for idx, idy in transform:
            new_kp.append(
                (prev_xs[idx-1], prev_ys[idy-1])
            )

        keypoint_annos[anno_info['image_id']] = new_kp

    print(os.path.abspath(os.path.dirname(__file__)))
    pred_heatmap = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)
    pred_coords = cal_coord(pred_heatmap, images_anno)
    # print('pred_coords ==>', pred_coords)
    scores = []
    for img_id in list(keypoint_annos.keys())[:]:
        groundtruth_anno = keypoint_annos[img_id]

        head_gt = groundtruth_anno[0]
        neck_gt = groundtruth_anno[1]

        threshold = math.sqrt((head_gt[0] - neck_gt[0]) ** 2 + (head_gt[1] - neck_gt[1]) ** 2)

        curr_score = []
        # for index, coord in enumerate(pred_coords[img_id]):
        # only use first 14 kps for prediction
        for index, coord in enumerate(pred_coords[img_id][:14]):
            pred_x, pred_y = coord
            gt_x, gt_y = groundtruth_anno[index]

            d = math.sqrt((pred_x-gt_x)**2 + (pred_y-gt_y)**2)
            if d > threshold:
                curr_score.append(0)
            else:
                curr_score.append(1)
        scores.append(np.mean(curr_score))

    print("PCKh=%.2f" % (np.mean(scores) * 100))

