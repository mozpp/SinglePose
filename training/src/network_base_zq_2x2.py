# -*- coding: utf-8 -*-
# @Time    : 18-4-24 5:48 PM
# @Author  : edvard_hua@live.com
# @FileName: network_base.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.contrib.slim as slim

_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_trainable = True


def is_trainable(trainable=True):
    global _trainable
    _trainable = trainable


def max_pool(inputs, k_h, k_w, s_h, s_w, name, padding="SAME"):
    return tf.nn.max_pool(inputs,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)


def upsample(inputs, factor, name, type='bilinear'):
    if type == 'bilinear':
        return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor], name=name)
    else:
        return tf.image.resize_nearest_neighbor(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor], name=name)
	

def separable_conv(input, c_o, k_s, stride, scope, padding='SAME'):
    with tf.variable_scope("%s"%scope):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.999,
                            fused=True,
                            is_training=_trainable,
                            activation_fn=tf.nn.relu6):
            output = slim.separable_convolution2d(input,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  trainable=_trainable,
                                                  depth_multiplier=1.0,
                                                  kernel_size=[k_s, k_s],
                                                  activation_fn=None,
                                                  weights_initializer=_init_xavier,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='dw',
                                                  padding=padding,
                                                  )

            output = slim.convolution2d(output,
                                        c_o,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=None,
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        trainable=_trainable,
                                        weights_regularizer=None,
                                        scope='sep',
                                        padding=padding,
                                        )

        return output


def inverted_bottleneck(inputs, up_channel_rate, channels, subsample, k_s=3, use_2x2=False, scope="", padding='SAME'):
    with tf.variable_scope("%s"%scope):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.999,
                            fused=True,
                            is_training=_trainable,
                            activation_fn=tf.nn.relu6):
            stride = 2 if subsample else 1

            output = slim.convolution2d(inputs,
                                        up_channel_rate * inputs.get_shape().as_list()[-1],
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=None,
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope='proj',
                                        trainable=_trainable,
                                        padding=padding,
                                        )

           # print("in 1 get_shape().as_list==>", output.get_shape().as_list)
            if not use_2x2:
                output = slim.separable_convolution2d(output,
                                                       num_outputs=None,
                                                       stride=stride,
                                                       depth_multiplier=1.0,
                                                       kernel_size=k_s,
                                                       activation_fn=None,
                                                       weights_initializer=_init_xavier,
                                                       weights_regularizer=_l2_regularizer_00004,
                                                       biases_initializer=None,
                                                       normalizer_fn=slim.batch_norm,
                                                       #padding="SAME",
                                                       scope='dw',
                                                       trainable=_trainable,
                                                       padding=padding,
                                                       )
            else:
                output = overall_block_2x2('SEPARABLE', output, 
                                 num_filters=up_channel_rate * inputs.get_shape().as_list()[-1],
                                 num_blocks=(k_s * k_s) // 8 - 1, 
                                 name="dw", 
                                 stride=None)
           # print("in 2 get_shape().as_list==>", output.get_shape().as_list)

            output = slim.convolution2d(output,
                                        channels,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=None,
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope='sep',
                                        trainable=_trainable,
                                        padding=padding,
                                        )
            if inputs.get_shape().as_list()[-1] == channels and stride == 1:
                output = tf.add(inputs, output)

    return output
	



def convb(input, k_h, k_w, c_o, stride, name, relu=True, padding='SAME'):
    with tf.variable_scope("%s"%name):
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
                activation_fn=tf.nn.relu6 if relu else None,
                scope="conv",
                trainable=_trainable,
                padding=padding,
                )
    return output
	
def dwconvb(input, k_h, k_w, stride, name, relu=True, padding='SAME'):
    with tf.variable_scope("%s"%name):
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=_trainable):
            output = slim.separable_convolution2d(
                inputs=input,
                num_outputs=None,
                kernel_size=[k_h, k_w],
                stride=stride,
                depth_multiplier=1.0,
                normalizer_fn=slim.batch_norm,
                weights_regularizer=_l2_regularizer_00004,
                weights_initializer=_init_xavier,
                biases_initializer=_init_zero,
                activation_fn=tf.nn.relu6 if relu else None,
                scope="dw",
                trainable=_trainable,
                padding=padding,
                )
    return output


def base_block_2x2(type, input, num_filters, name, add=True, relu=True):
    assert type in ['CONV', 'SEPARABLE']
    if type == 'CONV':
        fun = slim.convolution2d
    else:
        fun = slim.separable_convolution2d

    with tf.variable_scope("%s"%name):
        with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=_trainable):
            # pad 1
            padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            output = tf.pad(input, padding, 'CONSTANT')
            for i in range(2):
                # conv 2x2 2 times with no padding
                output = fun(
                    inputs=output,
                    num_outputs=num_filters if type == 'CONV' else None,
                    kernel_size=[2, 2],
                    stride=1,
                    normalizer_fn=slim.batch_norm,
                    #normalizer_fn=None,
                    weights_regularizer=_l2_regularizer_00004,
                    weights_initializer=_init_xavier,
                    biases_initializer=_init_zero,
                    activation_fn=tf.nn.relu6 if relu else None,
                    scope="{:s}2x2_{:d}".format(type, i),
                    trainable=_trainable,
                    padding='VALID')
            if add:
                return tf.add(input, output)
            else: 
                return output

def overall_block_2x2(type, input, num_filters, num_blocks, name, stride=(2, 2), relu=True):
    assert type in ['CONV', 'SEPARABLE']
    with tf.variable_scope("%s"%name):
        output = input
        for i in range(num_blocks):
            output = base_block_2x2(type, output, num_filters, str(i))
        
        if not stride is None:
            output = tf.nn.max_pool(output,
                                ksize=[1, 2, 2, 1],
                                strides=[1, stride[0], stride[1], 1],
                                padding='VALID',
                                name='maxpool')
        return output
