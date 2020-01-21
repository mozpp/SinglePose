import tensorflow as tf
import tensorflow.contrib.slim as slim

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_trainable = True

def hswish(inputs):
    return inputs * tf.nn.relu6(inputs + 3.) / 6.

def hsigmoid(inputs):
    return tf.nn.relu6(inputs + 3.) / 6.

def is_trainable(trainable=True):
    global _trainable
    _trainable = trainable


def max_pool(inputs, k_h, k_w, s_h, s_w, name, padding="SAME"):
    return tf.nn.max_pool(inputs,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)


def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)

def semodule(inputs, seratio, scope):
    height, width, channel = inputs.get_shape().as_list()[1], inputs.get_shape().as_list()[2], inputs.get_shape().as_list()[-1]
    output = slim.avg_pool2d(inputs, [height, width], stride=[height, width], scope=scope+'_se_avgpool')
    output = slim.convolution2d(
            inputs=output,
            num_outputs=int(channel * seratio),
            kernel_size=[1, 1],
            stride=1,
            normalizer_fn=None,
            weights_regularizer=None,
            weights_initializer=_init_xavier,
            biases_initializer=None,
            activation_fn=None,
            scope=scope + '_se_up',
            trainable=_trainable)
    output = tf.nn.relu(output)
    output = slim.convolution2d(
            inputs=output,
            num_outputs=channel,
            kernel_size=[1, 1],
            stride=1,
            normalizer_fn=None,
            weights_regularizer=None,
            weights_initializer=_init_xavier,
            biases_initializer=None,
            activation_fn=None,
            scope=scope + '_se_down',
            trainable=_trainable)
    output = hsigmoid(output)
    return output

def separable_conv(input, c_o, k_s, stride, scope, hs=True):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.999,
                            fused=True,
                            is_training=_trainable,
                            activation_fn=hswish if hs else tf.nn.relu6):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=_trainable,
                                                  depth_multiplier=1.0,
                                                  kernel_size=[k_s, k_s],
                                                  weights_initializer=_init_xavier,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  #normalizer_fn=slim.batch_norm,
                                                  scope=scope + '_depthwise')

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=_trainable,
                                        weights_regularizer=None,
                                        scope=scope + '_pointwise')

        return output


def inverted_bottleneck(inputs, up_channels, channels, subsample, k_s=3, hs=True, se=False, rate=1, scope=""):
    with tf.variable_scope("inverted_bottleneck_%s" % scope):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.999,
                            fused=True,
                            is_training=_trainable,
                            activation_fn=hswish if hs else tf.nn.relu6):
            stride = 2 if subsample else 1

            output = slim.convolution2d(inputs,
                                        up_channels,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope=scope + '_up_pointwise',
                                        trainable=_trainable)

            output = slim.separable_convolution2d(output,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1.0,
                                                  rate=rate,
                                                  kernel_size=k_s,
                                                  weights_initializer=_init_xavier,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  #normalizer_fn=slim.batch_norm,
                                                  padding="SAME",
                                                  scope=scope + '_depthwise',
                                                  trainable=_trainable)
            
            if se:
                seoutput = semodule(output, 0.25, scope)
                output = output * seoutput

            output = slim.convolution2d(output,
                                        channels,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=None,
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope=scope + '_pointwise',
                                        trainable=_trainable)
            if (not subsample) and inputs.get_shape().as_list()[-1] == channels:
                output = tf.add(inputs, output)

    return output

def convb(input, k_h, k_w, c_o, stride, name, hs=True):
    with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=_trainable):
        output = slim.convolution2d(
            inputs=input,
            num_outputs=c_o,
            kernel_size=[k_h, k_w],
            stride=stride,
            normalizer_fn=slim.batch_norm,
            weights_regularizer=_l2_regularizer_00004,
            weights_initializer=_init_xavier,
            biases_initializer=_init_zero,
            activation_fn=hswish if hs else tf.nn.relu6,
            scope=name,
            trainable=_trainable)
    return output
