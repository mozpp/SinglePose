# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from networks import get_network
import os

from pprint import pprint

os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
#parser.add_argument('--model', type=str, default='zq21_cpm_backbone_2x2')
#parser.add_argument('--model', type=str, default='zq21_cpm')
parser.add_argument('--model', type=str, default='yang_mv3_cpm')

parser.add_argument('--size', type=int, default=224)

parser.add_argument('--checkpoint', type=str, help='checkpoint path',
                    # default=r'../mv2_cpm_tiny/models/mv2_chk/model-458000',
                    # default=r'../mv2_cpm_tiny/models/mv2_chk/model-458000',
                    # default=r'../mv2_cpm_tiny/models/mv2_cpm_batch-14_lr-0.001_gpus-1_224x224_..-experiments-mv2_cpm/model-2000',
                    # default='../log/mv2_conn_overfitting/model/mv2_cpm_batch-1_lr-0.001_gpus-1_224x224_..-experiments-mv2_cpm_conn/model-44500'
                      default='/data/home/chenyu/Yangfeiyu/singlePose/log/zq_224/model/zq21_cpm_batch-96_lr-0.001_gpus-5_224x224_..-experiments-zq21_cpm/model-13000'
                    )
parser.add_argument('--output_node_names', type=str, default='CPM/stage_2_out')
# output_graph is not used here, pb file is generate under the same dir as chk file
parser.add_argument('--output_graph', type=str, default='./model.pb', help='output_freeze_path')

args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.size, args.size, 3], name="image")
# input_node = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="image")

with tf.Session() as sess:
    net = get_network(args.model, input_node, trainable=False)
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    input_graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )

chk_dir = '/' +  os.path.join(*args.checkpoint.split('/')[:-1])
pb_name = args.checkpoint.split('/')[-1] + '.pb'
args.output_graph = os.path.join(chk_dir, pb_name)

with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print("pb model is saved at: %s" % args.output_graph)
