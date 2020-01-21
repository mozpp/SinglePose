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
from pycocotools.coco import COCO

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


def cal_coord(pred_heatmaps, images_anno, args, predictions):
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
            if args.dataset_to_use == 'COCO':
                ori_img = cv2.imread(os.path.join(os.path.join(args.coco_dir, 'val2017'),
                                                images_anno[img_id]['file_name']))
            else:
                ori_img = cv2.imread(os.path.join(args.img_path, images_anno[img_id]['file_name']))

            img = vis_frame(ori_img, result)

            img_name = images_anno[img_id]['file_name']
            ind = img_name.find('.')
            img_saving_name = img_name[:ind] + '{:s}'.format('T' if predictions[img_id] else 'F') + img_name[ind:]
            cv2.imwrite(os.path.join(args.outputpath, img_saving_name), img)
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
    '''calc mask and ensure we get at least two maximums'''
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


def eval_accuracy(mean_diff, images_anno, threshold, verbose=True):
    ''' evaluate performance at given threshold '''
    num_imgs = len(mean_diff)

    # calc pred and gt
    pred_is_one_person = mean_diff > threshold
    gt_is_one_person = np.array([images_anno[img_id]['is_one_person'] for img_id in range(num_imgs)])

    # calc pred and acc
    predictions = gt_is_one_person == pred_is_one_person
    accuracy = np.sum(predictions) / num_imgs

    # print
    print('Threshold: %d, accuracy: %.2f' % (threshold, accuracy))
    if verbose:
        for img_id in range(num_imgs):
            print(images_anno[img_id]['file_name'], predictions[img_id])

    return predictions, accuracy

def get_coco_person_img_info(coco_dir, num_to_test=10):
    '''
    select some images suject to given contrains to build img info
    :param coco_dir: coco dataset path
    :param num_to_test: num of imgs to select
    :return: a dict with key being img file name and value being num of person
    '''
    coco_json_file = os.path.join(coco_dir, 'annotations/person_keypoints_val2017.json')

    cocoGt = COCO(coco_json_file)
    person_catId = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=person_catId)

    img_info = {}
    img_info_one_person = {}
    img_info_multi_persons = {}
    num_one_person = 0
    num_multi_person = 0
    while True:
        imgId = np.random.choice(imgIds, 1, replace=False)[0]

        # get annIds
        file_name = cocoGt.imgs[imgId]['file_name']
        img_height = cocoGt.imgs[imgId]['height']
        img_width = cocoGt.imgs[imgId]['width']

        annIds = cocoGt.getAnnIds([imgId])

        # exclude small imgs
        is_small = False
        for annId in annIds:
            ann = cocoGt.loadAnns([annId])[0]

            kps = ann['keypoints']
            ann_width = np.max(kps[0::3]) - np.min(kps[0::3])
            ann_height = np.max(kps[1::3]) - np.min(kps[1::3])

            if ann_height < img_height / 3 or ann_width < img_width / 3:
                is_small = True
                break
        if is_small:
            continue

        # get num of person whoes num of kps > 12
        num_person = 0
        for annId in annIds:
            ann = cocoGt.loadAnns([annId])[0]

            if ann['num_keypoints'] >= 14:
                num_person += 1

        # one person, half of num_to_test
        if num_person == 1 and num_one_person < num_to_test // 2:
            img_info_one_person[file_name] = num_person
            num_one_person += 1

        # 2 or 3 person, half of num_to_test
        if num_person in [2] and num_multi_person < num_to_test - num_to_test // 2:
            img_info_multi_persons[file_name] = num_person
            num_multi_person += 1

        # merge
        if num_one_person + num_multi_person >= num_to_test:
            img_info = {**img_info_one_person, **img_info_multi_persons}
            break

    return img_info

if __name__ == '__main__':
    pre()
    # set parameters
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--dataset_to_use", type=str, default="COCO", help="INPUT or COCO")
    parser.add_argument("--frozen_pb_path", type=str,
                        # default="../mv2_cpm_tiny/models/mv2_chk/mobilenetv2_model_592000_224x224_tf1.14.pb",
                        default="../mv2_cpm_tiny/models/mv2_chk/mv2_chk-320x320.pb",
                        )
    parser.add_argument('--coco_dir', type=str, default='/media/yangfeiyu/560dccd6-4f69-40e3-9d5b-c5d93d907f80/yangfeiyu/DataSets/COCO/COCO-2017')
    parser.add_argument("--img_path", type=str, default="../demo_io/inputs")
    parser.add_argument("--output_node_name", type=str, default="Convolutional_Pose_Machine/stage_5_out")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--save_imgs", type=bool, default=True)
    parser.add_argument("--outputpath", type=str, default="../demo_io/coco_outputs")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    assert args.dataset_to_use in ['INPUT', 'COCO']

    # reset output folder
    if os.path.exists(args.outputpath):
        shutil.rmtree(args.outputpath)
    os.makedirs(args.outputpath)

    # create annotation
    images_anno = {}
        # get imgIds from COCO
    if args.dataset_to_use == 'COCO':
        coco_img_info = get_coco_person_img_info(args.coco_dir, num_to_test=100)

        ind = 0
        print("Total test example=%d" % len(coco_img_info))
        for img_file_name, num_person in coco_img_info.items():
            img = cv2.imread(os.path.join(args.coco_dir, 'val2017', img_file_name))
            img_info = {}
            img_info['file_name'] = img_file_name
            img_info['height'] = img.shape[0]
            img_info['width'] = img.shape[1]
            img_info['id'] = ind
            img_info['is_one_person'] = False if num_person != 1 else True
            images_anno[ind] = img_info
            ind += 1
        # get imgIds from INPUT
    else:
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

    # predict
    if args.dataset_to_use == 'COCO':
        pred_heatmap = infer(args.frozen_pb_path, args.output_node_name,
                            os.path.join(args.coco_dir, 'val2017'), images_anno)
    else:
        pred_heatmap = infer(args.frozen_pb_path, args.output_node_name, args.img_path, images_anno)

    # normalize hm
    normalized_hm = pred_heatmap.copy()
    for k, v in normalized_hm.items():
        max_val = np.max(v)
        normalized_hm[k] = 255. * v / max_val

    # initialize diff between max and second max
    num_imgs = len(normalized_hm.keys())
    num_kps = normalized_hm[1].shape[-1]
    diff = np.ones([num_imgs, num_kps]) * -1

    # calc diff
    for joint in range(num_kps):
        # create saving folders if needed
        if args.save_imgs:
            joint_file_name = os.path.join(args.outputpath, str(joint))
            if os.path.exists(joint_file_name):
                shutil.rmtree(joint_file_name)
            os.makedirs(joint_file_name)

        # calc diff between max and second max
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

    # evaluate performance
    best_acc = 0
    best_th = None
    best_pred = None

    thresholds = np.arange(0,255,2)
    for th in thresholds:
        # print('================= threshold: {:d} ===================='.format(th))
        predictions, accuracy = eval_accuracy(mean_diff, images_anno, th, verbose=False)
        if accuracy > best_acc:
            best_acc = accuracy
            best_pred = predictions
            best_th = th

    final_result = cal_coord(pred_heatmap, images_anno, args, best_pred)
    write_json(final_result, args.outputpath)