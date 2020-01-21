import tensorflow as tf
import tensorflow.contrib.slim as slim

from network_base_mv3 import max_pool, upsample, inverted_bottleneck, separable_conv, convb, is_trainable

N_KPOINTS = 14
STAGE_NUM = 6

ratio = 1.
out_channel_ratio = lambda d: max(int(d * ratio), 8)
up_channel_ratio = lambda d: max(int(d * ratio), 8)
out_channel_cpm = lambda d: max(int(d * ratio), 8)


def build_network(input, trainable):
    is_trainable(trainable)

    net = convb(input, 3, 3, out_channel_ratio(16), 2, name="Conv2d_0")

    with tf.variable_scope('MobilenetV3'):

        # 128, 112
        mv3_branch_0 = inverted_bottleneck(net, up_channel_ratio(16), out_channel_ratio(16), 1, 3, False, True, 1, scope="MobilenetV3_part_0")

        mv3_branch_1 = inverted_bottleneck(mv3_branch_0, up_channel_ratio(72), out_channel_ratio(24), 1, 3, False, False, 1, scope="MobilenetV3_part_1")

        mv3_branch_2 = inverted_bottleneck(mv3_branch_1, up_channel_ratio(88), out_channel_ratio(24), 0, 3, False, False, 1, scope="MobilenetV3_part_2")

        mv3_branch_3 = inverted_bottleneck(mv3_branch_2, up_channel_ratio(96), out_channel_ratio(40), 0, 5, True, True, 1, scope="MobilenetV3_part_3")

        mv3_branch_4 = inverted_bottleneck(mv3_branch_3, up_channel_ratio(240), out_channel_ratio(40), 0, 5, True, True, 2, scope="MobilenetV3_part_4")

        mv3_branch_5 = inverted_bottleneck(mv3_branch_4, up_channel_ratio(240), out_channel_ratio(40), 0, 5, True, True, 1, scope="MobilenetV3_part_5")

        mv3_branch_6 = inverted_bottleneck(mv3_branch_5, up_channel_ratio(120), out_channel_ratio(48), 0, 5, True, True, 1, scope="MobilenetV3_part_6")

        mv3_branch_7 = inverted_bottleneck(mv3_branch_6, up_channel_ratio(144), out_channel_ratio(48), 0, 5, True, True, 1, scope="MobilenetV3_part_7")

        mv3_branch_8 = inverted_bottleneck(mv3_branch_7, up_channel_ratio(288), out_channel_ratio(96), 1, 5, True, True, 1, scope="MobilenetV3_part_8")

        mv3_branch_9 = inverted_bottleneck(mv3_branch_8, up_channel_ratio(576), out_channel_ratio(96), 0, 5, True, True, 1, scope="MobilenetV3_part_9")

        mv3_branch_10 = inverted_bottleneck(mv3_branch_9, up_channel_ratio(576), out_channel_ratio(96), 0, 5, True, True, 1, scope="MobilenetV3_part_10")

        cancat_mv2 = tf.concat(
            [
                max_pool(mv3_branch_0, 2, 2, 2, 2, name="mv3_0_max_pool"),
                mv3_branch_7,
                upsample(mv3_branch_10, 2, name="mv3_4_upsample")
            ]
            , axis=3)

    with tf.variable_scope("Convolutional_Pose_Machine"):
        l2s = []
        prev = None
        for stage_number in range(STAGE_NUM):
            if prev is not None:
                inputs = tf.concat([cancat_mv2, prev], axis=3)
            else:
                inputs = cancat_mv2

            kernel_size = 7
            lastest_channel_size = 128
            if stage_number == 0:
                kernel_size = 3
                lastest_channel_size = 512
            
            inputs_channel = inputs.get_shape().as_list()[-1]
            '''
            if stage_number == 0:
                _ = slim.stack(inputs, inverted_bottleneck,
                            [
                                (up_channel_ratio(inputs_channel * 2), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                                (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                                (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                            ], scope="stage_%d_mv2" % stage_number)

                _ = slim.stack(_, separable_conv,
                            [
                                (out_channel_ratio(lastest_channel_size), 1, 1),
                                (N_KPOINTS, 1, 1)
                            ], scope="stage_%d_mv1" % stage_number)
            else:
                _ = slim.stack(inputs, inverted_bottleneck,
                            [
                                (up_channel_ratio(inputs_channel * 2), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                                (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                                (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                                (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                                (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size, False, False, 1),
                            ], scope="stage_%d_mv2" % stage_number)

                _ = slim.stack(_, separable_conv,
                            [
                                (out_channel_ratio(lastest_channel_size), 1, 1),
                                (N_KPOINTS, 1, 1)
                            ], scope="stage_%d_mv1" % stage_number)
            '''
            _ = slim.stack(inputs, inverted_bottleneck,
                           [
                               (up_channel_ratio(inputs_channel * 2), out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(128), out_channel_cpm(32), 0, kernel_size),
                           ], scope="stage_%d_mv2" % stage_number)

            _ = slim.stack(_, separable_conv,
                           [
                               (out_channel_ratio(lastest_channel_size), 1, 1),
                               (N_KPOINTS, 1, 1)
                           ], scope="stage_%d_mv1" % stage_number)
            prev = _
            cpm_out = upsample(_, 4, "stage_%d_out" % stage_number)
            l2s.append(cpm_out)

    return cpm_out, l2s
