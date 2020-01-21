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
from pPose_nms import write_json
from fn import vis_frame

parts = ["top_head", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
         "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]


def pose_processing(hm, coord):
    px = int(math.floor(coord[0] + 0.5))
    py = int(math.floor(coord[1] + 0.5))
    if 1 < px < hm.shape[0]-1 and 1 < py < hm.shape[1]-1:
        diff = np.array([hm[py][px+1] - hm[py][px-1], hm[py+1][px] - hm[py-1][px]])
        coord += np.sign(diff) * 0.25

    coord = np.array(coord)
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
            'keypoints':coord,
            'kp_score':scores})
        result = {
            'imgname': images_anno[img_id]['file_name'],
            'result': result
        }
        if args.save_imgs:
            ori_img = cv2.imread(os.path.join(args.img_path, images_anno[img_id]['file_name']))
            img = vis_frame(ori_img, result)
            cv2.imshow('demo', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(os.path.join(args.outputpath, 'vis', images_anno[img_id]['file_name']), img)
        final_result.append(result)
    return final_result

def plot_heat(img, heat):
    import matplotlib.pyplot as plt
    heat = np.squeeze(heat)
    img = np.squeeze(img)

    h, w, c = heat.shape
    assert c in [14, 33]

    if c == 14:
        fig, axs = plt.subplots(3, 5)
        axs = np.reshape(axs, [-1])
        for i in range(c):
            axs[i].imshow(heat[:,:,i], plt.cm.gray)
            axs[i].set_title(parts[i], fontsize=10)
        axs[-1].imshow(img[:,:,::-1])

    if c == 33:
        fig, axs = plt.subplots(5, 7, figsize=[100,100])
        axs = np.reshape(axs, [-1])
        for i in range(c):
            axs[i].imshow(heat[:,:,i], plt.cm.gray)
            if i < len(parts):
                axs[i].set_title(parts[i], fontsize=10)
            elif i < len(parts) + 13:
                axs[i].set_title(i - 14, fontsize=10)
        axs[-2].imshow(img[:,:,::-1])
        # axs[-1].imshow(np.max(heat, axis=-1))
        axs[-1].imshow(np.max(heat[:,:,:14], axis=-1))


    for ax in axs:
        ax.axis('off')

    plt.show()



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

            print(heat.shape)
            for i in range(33):
                print(i, '==>', np.max(heat[0,:,:,i]))
            # exit()

            plot_heat(inp_img, heat)

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

if __name__ == '__main__':
    pre()

    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--frozen_pb_path", type=str,
                        # default="../mv2_cpm_tiny/models/mv2_chk/mobilenetv2_model_592000_224x224_tf1.14.pb",
                        # default="../mv2_cpm_tiny/models/mv2_chk/mv2_chk-320x320.pb",

                        # good overfitting conn model, loss=470
                        # default="../log/mv2_conn_overfitting/model/mv2_cpm_batch-1_lr-0.001_gpus-1_224x224_..-experiments-mv2_cpm_conn/model-44500.pb"
                        default="../log/mv2_conn_224/from_220/mv2_conn_224/model/mv2_cpm_33_batch-64_lr-0.001_gpus-3_224x224_..-experiments-mv2_cpm_conn/model-116000.pb"
                        )
    parser.add_argument("--img_path", type=str, default="../demo_io/overfitting")
    parser.add_argument("--output_node_name", type=str, default="Convolutional_Pose_Machine/stage_5_out")
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--save_imgs", type=bool, default=True)
    parser.add_argument("--outputpath", type=str, default="../demo_io/outputs")
    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    #anno = json.load(open(args.anno_json_path))


    # test with imgs from dataset
    from pycocotools.coco import COCO
    img_path = '../dataset'
    args.img_path = img_path

    TRAIN_JSON = "ai_challenger_train.json"
    VALID_JSON = "ai_challenger_valid.json"
    COCO_SINGLE_JSON = "coco_train.json"

    coco = COCO(os.path.join(img_path, 'aich_plus_coco', VALID_JSON))
    imgIds = coco.getImgIds()

    while True:
        imgId = np.random.choice(imgIds, 1, replace=False)[0]
        img_meta = coco.loadImgs([imgId])[0]
        img_file_name = img_meta['file_name']
        full_img_file_name = os.path.join(img_path, img_file_name)
        print('img path ==> %s' % full_img_file_name)

        images_anno = {}
        img = cv2.imread(full_img_file_name)

        img_info = {}
        img_info['file_name'] = img_meta['file_name']
        img_info['height'] = img.shape[0]
        img_info['width'] = img.shape[1]
        img_info['id'] = imgId
        img_info['is_one_person'] = True
        images_anno[0] = img_meta






        # # test with imgs from input file
        # images_anno = {}
        # filelist = os.listdir(args.img_path)
        # # filelist.sort(key=lambda x: int(x[:-4]))
        # print("Total test example=%d" % len(filelist))
        # for index, item in enumerate(filelist):
        #     img = cv2.imread(os.path.join(args.img_path, filelist[index]))
        #     img_info = {}
        #     img_info['file_name'] = item
        #     img_info['height'] = img.shape[0]
        #     img_info['width'] = img.shape[1]
        #     img_info['id'] = index
        #     img_info['is_one_person'] = False if len(item) == 16 else True
        #     images_anno[index] = img_info

        pred_heatmap = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)

    final_result = cal_coord(pred_heatmap, images_anno, args)
    write_json(final_result, args.outputpath)