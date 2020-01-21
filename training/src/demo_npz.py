# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import json
import argparse
import cv2
import os
import math
import time

from scipy.ndimage.filters import gaussian_filter
from pPose_nms import write_json
from fn import vis_frame

def pose_processing(hm, coord):
    px = int(math.floor(coord[0] + 0.5))
    py = int(math.floor(coord[1] + 0.5))
    if 1 < px < hm.shape[0]-1 and 1 < py < hm.shape[1]-1:
        diff = np.array([hm[py][px+1] - hm[py][px-1], hm[py+1][px] - hm[py-1][px]])
        coord += np.sign(diff) * 0.25
    return coord
'''
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
            #heat1 = heat - np.max(heat)
            #heat1 = np.exp(heat1) / np.sum(np.exp(heat1))
            scores.append(np.max(heat))
            heat = gaussian_filter(heat, sigma=5)
            ind = np.unravel_index(np.argmax(heat), heat.shape)
            ind = pose_processing(heat, ind)
            coord_x = int((ind[1] + 1) / scale_w)
            coord_y = int((ind[0] + 1) / scale_h)
            coord.append((coord_x, coord_y))
        result.append({
            'keypoints':coord,
            'kp_score':scores})
        result = {
            'imgname': images_anno[img_id]['file_name'],
            'result': result
        }
        if args.save_imgs:
            ori_img = cv2.imread(os.path.join(args.img_path, images_anno[img_id]['file_name']))
            img = vis_frame(ori_img, result)
            #cv2.imshow("result", img)
            #cv2.waitKey(0)
            cv2.imwrite(os.path.join(args.output_path, 'vis', images_anno[img_id]['file_name']), img)
        final_result.append(result)
    return final_result
'''

def cal_coord(pred_heatmaps, images_anno, args):
    keypoint_transforms = [[0, 10], [1, 8], [2, 14], [3, 15], [4, 16], [5, 11], [6, 12], [7, 13],
                         [8, 1], [9, 2], [10, 3], [11, 4], [12, 5], [13, 6]]
    final_result = []
    for img_id in pred_heatmaps.keys():
        heat_h, heat_w, n_kpoints = pred_heatmaps[img_id].shape
        scale_h, scale_w = heat_h / images_anno[img_id]['height'], heat_w / images_anno[img_id]['width']
        coord = []
        scores = []
        result = np.zeros((17, 2))
        for p_ind in range(n_kpoints):
            heat = pred_heatmaps[img_id][:, :, p_ind]
            #heat1 = heat - np.max(heat)
            #heat1 = np.exp(heat1) / np.sum(np.exp(heat1))
            scores.append(np.max(heat))
            heat = gaussian_filter(heat, sigma=5)
            ind = np.unravel_index(np.argmax(heat), heat.shape)
            ind = pose_processing(heat, ind)
            coord_x = int((ind[1] + 1) / scale_w)
            coord_y = int((ind[0] + 1) / scale_h)
            coord.append([coord_x, coord_y])
        for keypoint_transform in keypoint_transforms:
            result[keypoint_transform[1]] = coord[keypoint_transform[0]]
        result[9] = (result[8] + result[10]) / 2
        result[0] = (result[1] + result[4]) / 2
        result[7] = (result[0] + result[8]) / 2
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PCKh benchmark")
    parser.add_argument("--frozen_pb_path", type=str, default="/workspace/release/cpm_model/model458000_224_7stage.pb")
    parser.add_argument("--img_path", type=str, default="/workspace/Dataset/Astra/Normal/2-2.5m")
    parser.add_argument("--output_node_name", type=str, default="Convolutional_Pose_Machine/stage_5_out")
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--save_imgs", type=bool, default=True)
    parser.add_argument("--output_path", type=str, default="/workspace/Dataset/Astra/Normal")
    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    #anno = json.load(open(args.anno_json_path))

    images_anno = {}
    filelist = os.listdir(args.img_path)
    filelist.sort(key= lambda x:int(x[:-4]))
    print("Total test example=%d" % len(filelist))
    for index, item in enumerate(filelist):
        img_info = {}
        img_info['file_name'] = item
        img_info['height'] = 480
        img_info['width'] = 640
        img_info['id'] = index
        images_anno[index] = img_info

    pred_heatmap = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)
    final_result = cal_coord(pred_heatmap, images_anno, args)

    dictionarry_keypoints={'S1': {'Directions 1' : np.asarray([final_result])}}
    metadata = {'layout_name': 'h36m', 'num_joints': 17, 'keypoints_symmetry': [[4, 5, 6, 11, 12, 13],[1, 2, 3, 14, 15, 16]]}
    #np.savez(os.path.join('/home/narvis/Dev/VideoPose3D/data', "data_2d_detections.npz"), metadata=metadata, positions_2d=dictionarry_keypoints)
    np.savez(os.path.join(args.output_path, "data_2d_detections_mobilenet_224_normal_2.npz"), metadata=metadata, positions_2d=dictionarry_keypoints)
    
    #write_json(final_result, args.output_path)