import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# print(sys.path)

# from mobilenet import mobilenet_v3
from mobilenet import mobilenet_v3_with_2x2 as mobilenet_v3
from network_base_zq import max_pool, upsample, inverted_bottleneck, \
                            separable_conv, convb, dwconvb, is_trainable

N_KPOINTS = 14
STAGE_NUM = 7


def build_network(input, trainable):
    is_trainable(trainable)

    with tf.variable_scope('MV'):
        logits, endpoints = mobilenet_v3.mobilenet(
            input,
            conv_defs=mobilenet_v3.V3_SMALL_MINIMALISTIC_2x2,
            base_only=True)

        # # print names of all nodes
        # print_default_graph_def()

        # # show endpoints
        # for k, v in endpoints.items():
        #     print(k, v)

        # get tensors of each stage
        output_112_16 = endpoints['layer_1']
        output_56_16 = endpoints['layer_2']
        output_28_24 = endpoints['layer_4']
        output_14_48 = endpoints['layer_9']
        output_7_96 = endpoints['layer_12']

        # # get tensors of each stage by name
        # graph = tf.get_default_graph()
        # cur_scp = graph.get_name_scope() + '/'
        #
        # output_112_16 = graph.get_tensor_by_name(cur_scp+'MobilenetV3/Conv/Relu:0')
        # output_56_16 = graph.get_tensor_by_name(cur_scp+'MobilenetV3/expanded_conv/output:0')
        # output_28_24 = graph.get_tensor_by_name(cur_scp+'MobilenetV3/expanded_conv_2/output:0')
        # output_14_48 = graph.get_tensor_by_name(cur_scp+'MobilenetV3/expanded_conv_7/output:0')
        # output_7_96 = graph.get_tensor_by_name(cur_scp+'MobilenetV3/expanded_conv_10/output:0')

        cancat_mv2 = tf.concat(
            [
                max_pool(output_112_16, 4, 4, 4, 4, name="mv2_0_max_pool"),
                max_pool(output_56_16, 2, 2, 2, 2, name="mv2_1_max_pool"),
                output_28_24,
                upsample(output_14_48, 2, name="mv2_3_upsample"),
                upsample(output_7_96, 4, name="mv2_4_upsample")
            ],
            axis=3)

    with tf.variable_scope("CPM"):
        l2s = []
        prev = None
        for stage_number in range(STAGE_NUM):
            if prev is not None:
                inputs = tf.concat([cancat_mv2, prev], axis=3)
            else:
                inputs = cancat_mv2

            # kernel_size = 7
            lastest_channel_size = 384
            if stage_number == 0:
                #    kernel_size = 7
                lastest_channel_size = 128

            _ = slim.stack(inputs, inverted_bottleneck,
                           [
                               (1, 32, 0, 7),
                               (1, 32, 0, 7),
                               # (1, 32, 0, 3),
                               # (1, 32, 0, 3),
                               # (1, 32, 0, 3),
                               # (1, 32, 0, 3),
                           ], scope="stage_%d_mv2" % stage_number)

            _ = slim.stack(_, separable_conv,
                           [
                               (lastest_channel_size, 1, 1),
                               (N_KPOINTS, 1, 1)
                           ], scope="stage_%d_mv1" % stage_number)

            prev = _
            cpm_out = upsample(_, 4, "stage_%d_out" % stage_number)
            l2s.append(cpm_out)
    return cpm_out, l2s

def print_default_graph_def():
    graph_def = tf.get_default_graph().as_graph_def()
    print(graph_def)

def save_frozen_model(output_node_name, pb_file_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = \
            tf.graph_util.convert_variables_to_constants(sess,
                                                         input_graph_def,
                                                         output_node_name.split(","))
    with tf.gfile.GFile(pb_file_name, 'w') as f:
        f.write(output_graph_def.SerializeToString())

def pre():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)

if __name__ == '__main__':
    pre()
    test_mobilenet_v3 = False
    test_build_network_func = True

    # test mv3
    if test_mobilenet_v3:
        logits, endpoints = mobilenet_v3.mobilenet(
            tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32),
            conv_defs=mobilenet_v3.V3_SMALL_MINIMALISTIC,
            base_only=True)

        # print names of all nodes
        print_default_graph_def()

        # print endpoints
        sorted_list = sorted(endpoints.items(), key=lambda x: x[0])
        for i, v in enumerate(sorted_list):
            print(i, v)

        # save pb file
        save_frozen_model('MobilenetV3/expanded_conv_10/output', 'V3_SMALL_MINIMALISTIC_base_only.pb')

    # test build_network function
    if test_build_network_func:
        input = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32, name="image")
        cpm_out, l2s = build_network(input,
                                     trainable=True)

        # print names of all nodes
        print_default_graph_def()

        # save pb file
        save_frozen_model('CPM/stage_2_out', 'V3_SMALL_MINIMALISTIC_CPM.pb')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cpm_out_, l2s_ = sess.run([cpm_out, l2s],
                                      feed_dict={input: np.random.random(size=[1,224,224,3])})
            print(cpm_out_.shape, [x.shape for x in l2s_])