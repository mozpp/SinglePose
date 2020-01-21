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

def pre():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)


def pose_processing(hm, coord):
    px = int(math.floor(coord[0] + 0.5))
    py = int(math.floor(coord[1] + 0.5))
    if 1 < px < hm.shape[0]-1 and 1 < py < hm.shape[1]-1:
        diff = np.array([hm[py][px+1] - hm[py][px-1], hm[py+1][px] - hm[py-1][px]])
        coord += np.sign(diff) * 0.25
    return coord

if __name__ == '__main__':
    pre()
    parser = argparse.ArgumentParser(description="PCKh benchmark")
    parser.add_argument("--frozen_pb_path", type=str,
                        # default="../mv3_cpm_tiny/models/mv3_chk/mv3_chk.pb",
                        # default="../mv2_cpm_tiny/models/mv2_chk/mv2_chk-320x320.pb",
                        #  default="../log/mv2_224/from_95_yuge_best/model-592000.pb",
                        default="../log/zq21_model_70000_224x224_2stage_tf1.14.pb",

                        )
    # parser.add_argument("--video_path", type=str, default="../video/yang_video/潘哲-时代在召唤.mp4")
    parser.add_argument("--video_path", type=str, default="/media/yangfeiyu/560dccd6-4f69-40e3-9d5b-c5d93d907f80/yangfeiyu/Astra/Normal/2-2.5m/originalData/Video/chenyu2.avi")


    # parser.add_argument("--output_node_name", type=str, default="Convolutional_Pose_Machine/stage_5_out")
    parser.add_argument("--output_node_name", type=str, default="CPM/stage_5_out")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--save_imgs", type=bool, default=True)
    # parser.add_argument("--outputpath", type=str, default="../video/output_video")
    parser.add_argument("--outputpath", type=str, default="/media/yangfeiyu/560dccd6-4f69-40e3-9d5b-c5d93d907f80/yangfeiyu/Astra/Normal/2-2.5m/originalData/Video_output/")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.gfile.GFile(args.frozen_pb_path, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name="")
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name("image:0")
    output_heat = graph.get_tensor_by_name("%s:0" % args.output_node_name)

    # add subdir
    video_name = args.video_path.split('/')[-1]
    output_subdir = os.path.join(args.outputpath, video_name.split('.')[0])
    if os.path.exists(output_subdir):
        shutil.rmtree(output_subdir)
    os.makedirs(output_subdir)
    args.outputpath = output_subdir

    final_result = []
    total_img = 0
    with tf.Session() as sess:
        video_capture = cv2.VideoCapture(args.video_path)

        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        # for avi
        fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
        # for mp4
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout_1 = cv2.VideoWriter(os.path.join(args.outputpath, video_name), fourcc, fps, (width, height))

        start_second = 0
        start_frame = fps * start_second
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while True:
            retval, img_data = video_capture.read()

            if not retval:
                break
            #img_data = np.rot90(img_data, -1)
            image_shape = img_data.shape
            shape = input_image.get_shape().as_list()
            inp_img = cv2.resize(img_data, (shape[1], shape[2]))
            st = time.time()
            heatmap = sess.run(output_heat, feed_dict={input_image: [inp_img]})
            infer_time = 1000 * (time.time() - st)
            total_img+=1
            print("img_id = %d, cost_time = %.2f ms" % (total_img, infer_time))
            #cv2.imshow('result', img_data)
            #cv2.waitKey(1)
            _, heat_h, heat_w, n_kpoints = heatmap.shape
            imgname = str(total_img) + 'jpg'
            '''
            resize_heatmap=tf.image.resize_bilinear(heatmap, [int(input_image.get_shape()[1]) , int(input_image.get_shape()[2]) ])
            
           # _,heat_h, heat_w, n_kpoints = resize_heatmap.shape
            _, heat_h, heat_w, n_kpoints=resize_heatmap.get_shape().as_list()
            image_h,image_w,ch=img_data.shape

            scale_h = float(heat_h) / image_h
            scale_w =float(heat_w) / image_w
            '''
            scale_h, scale_w = heat_h / image_shape[0], heat_w / image_shape[1]
            coord = []
            scores = []
            result = []
            for p_ind in range(n_kpoints):
                heat = heatmap[0,:, :, p_ind]
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
                'imgname': imgname,
                'result': result
            }

            if args.save_imgs:
                img = vis_frame(img_data, result)
                vout_1.write(img)

                # cv2.imwrite(os.path.join(args.outputpath, video_name.split('.')[0] + str(total_img) + '.jpg'), img)
                inp_img = cv2.resize(img, (int(image_shape[1]*0.5), int(image_shape[0]*0.5)))
                cv2.imshow('img', inp_img)
                cv2.waitKey(1)
            final_result.append(result)
        write_json(final_result, args.outputpath)
        vout_1.release()


