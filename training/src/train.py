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
import tensorflow.contrib.slim as slim
import os
import platform
import time
import numpy as np
import configparser
import dataset
import json

from datetime import datetime
from amsgrad import AMSGrad

from dataset import get_train_dataset_pipeline, get_valid_dataset_pipeline
from networks import get_network
from dataset_prepare import CocoPose
from dataset_augment import set_network_input_wh, set_network_scale

import sys
from logger import Logger
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('log.txt')


""""""
from losses.yang_losses import symmetrical_loss, distance_loss, custom_l2_loss
""""""


def update_imgId_loss_dict(imgId_loss_dict, imgIds_, loss_of_ids_):
    for i in range(len(imgIds_)):
        key = imgIds_[i]
        if imgId_loss_dict.get(key) == None:
            imgId_loss_dict[key] = loss_of_ids_[i]
        else:
            imgId_loss_dict[key] += loss_of_ids_[i]
            imgId_loss_dict[key] /= 2.


# def get_loss_and_output(model, batchsize, input_image, input_heat, reuse_variables=None):
#     losses = []
#
#     with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
#         _, pred_heatmaps_all = get_network(model, input_image, True)
#
#     for idx, pred_heat in enumerate(pred_heatmaps_all):
#         # loss_l2 = focal_loss_sigmoid(tf.concat(pred_heat, axis=0), input_heat)
#         # loss_l2 = tf.nn.l2_loss(tf.concat(pred_heat, axis=0) - input_heat, name='loss_heatmap_stage%d' % idx)
#         loss_l2 = custom_l2_loss(pred_heat, input_heat)
#
#         losses.append(loss_l2)
#
#     total_loss = tf.reduce_sum(losses) / batchsize
#     total_loss_ll_heat = tf.reduce_sum(loss_l2) / batchsize
#     loss_of_ids = loss_l2
#     return total_loss, total_loss_ll_heat, pred_heat, loss_of_ids

def get_loss_and_output(model, batchsize, input_image, input_heat, reuse_variables=None):
    losses_l2 = []
    losses_symm = []
    losses_dist = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        _, pred_heatmaps_all = get_network(model, input_image, True)

    for idx, pred_heat in enumerate(pred_heatmaps_all):
        # loss_l2 = focal_loss_sigmoid(tf.concat(pred_heat, axis=0), input_heat)
        # loss_l2 = tf.nn.l2_loss(tf.concat(pred_heat, axis=0) - input_heat, name='loss_heatmap_stage%d' % idx)
        loss_l2 = custom_l2_loss(pred_heat, input_heat)

        loss_l2_scalar = tf.reduce_sum(loss_l2)
        loss_symm = symmetrical_loss(pred_heat, [input_image])[0]
        loss_dist = distance_loss(pred_heat)

        losses_l2.append(loss_l2_scalar)
        losses_symm.append(loss_symm)
        losses_dist.append(loss_dist)


    l_l2 = tf.reduce_sum(losses_l2) / batchsize
    l_symm = tf.reduce_sum(losses_symm) / batchsize * 1e-5
    l_dist = tf.reduce_sum(losses_dist) / batchsize * 1e2

    #total_loss = l_l2 + l_symm + l_dist
    total_loss = l_l2 #+ l_symm

    total_loss_ll_heat = tf.reduce_sum(loss_l2) / batchsize
    loss_of_ids = loss_l2
    return total_loss, total_loss_ll_heat, pred_heat, loss_of_ids, l_l2, l_symm, l_dist

def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def pre():
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)

def main(argv=None):
    #pre()

    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config_file = "training/experiments/zq21_cpm.cfg"
    if len(argv) != 1:
        config_file = argv[1]
    config.read(config_file)
    for _ in config.options("Train"):
        params[_] = eval(config.get("Train", _))

    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']

    gpus_index = params['visible_devices'].split(",")
    params['gpus'] = len(gpus_index)

    if not os.path.exists(params['modelpath']):
        os.makedirs(params['modelpath'])
    if not os.path.exists(params['logpath']):
        os.makedirs(params['logpath'])

    dataset.set_config(params)
    set_network_input_wh(params['input_width'], params['input_height'])
    set_network_scale(params['scale'])

    gpus = 'gpus'
    if platform.system() == 'Darwin':
        gpus = 'cpu'
    training_name = '{}_batch-{}_lr-{}_{}-{}_{}x{}_{}'.format(
        params['model'],
        params['batchsize'],
        params['lr'],
        gpus,
        params['gpus'],
        params['input_width'], params['input_height'],
        config_file.replace("/", "-").replace(".cfg", "")
    )

    # start for loss json
    use_loss_json = params['use_loss_json']
    imgid_loss_file = params['imgid_loss_file']
    input_imgid_loss_dict = None
    if use_loss_json:
        with open(imgid_loss_file, 'r') as f:
            input_imgid_loss_dict = json.load(f)
            print("JSON READ !")
    # end for loss json


    with tf.Graph().as_default(), tf.device("/cpu:0"):
    #with tf.Graph().as_default():
        train_dataset = get_train_dataset_pipeline(input_imgid_loss_dict, params['batchsize'], params['max_epoch'], buffer_size=100)
        valid_dataset = get_valid_dataset_pipeline(params['batchsize'], params['max_epoch'], buffer_size=100)

        train_iterator = train_dataset.make_one_shot_iterator()
        valid_iterator = valid_dataset.make_one_shot_iterator()
        
        handle = tf.placeholder(tf.string, shape=[])
        input_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

        global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
                                                   decay_steps=7500, decay_rate=float(params['decay_rate']), staircase=True)


        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        #opt = AMSGrad(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        tower_grads = []
        reuse_variable = False

        if platform.system() == 'Darwin':
            # cpu (mac only)
            with tf.device("/cpu:0"):
                with tf.name_scope("CPU_0"):
                    input_image, input_heat, imgIds = input_iterator.get_next()
                    loss, last_heat_loss, pred_heat, loss_of_ids, l_l2, l_symm, l_dist = get_loss_and_output(params['model'], params['batchsize'],
                                                                          input_image, input_heat, reuse_variable)
                    reuse_variable = True
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
        else:
            # multiple gpus
            for i in range(params['gpus']):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("GPU_%d" % i):
                        input_image, input_heat, imgIds = input_iterator.get_next()  # 取出dataset-meta数据
                        loss, last_heat_loss, pred_heat, loss_of_ids, l_l2, l_symm, l_dist = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
                        reuse_variable = True
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(max_to_keep=100)

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_l2", l_l2)
        tf.summary.scalar("loss_symm", l_symm)
        tf.summary.scalar("loss_dist", l_dist)
        tf.summary.scalar("loss_lastlayer_heat", last_heat_loss)
        summary_merge_op = tf.summary.merge_all()

        pred_result_image = tf.placeholder(tf.float32, shape=[params['batchsize'], 480, 640, 3])
        pred_result__summary = tf.summary.image("pred_result_image", pred_result_image, params['batchsize'])

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            if params['checkpoint'] != False:
                
                print('restore models param')
                model_path = params['checkpoint']
                print('model_path: ', model_path)
                model_dict = '/'.join(model_path.split('/')[:-1])
                ckpt = tf.train.get_checkpoint_state(model_dict)
                readstate = ckpt and ckpt.model_checkpoint_path
                # assert readstate, "the params dictionary is not valid"
                #saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))

                # variables_to_restore = slim.get_variables_to_restore()
                # slim.assign_from_checkpoint(model_path, variables_to_restore, ignore_missing_vars=True)
                saver.restore(sess, model_path)
                
                '''
                model_path = params['checkpoint']
                print('model_path: ', model_path)
                init_assign_op, init_feed_dict = slim.assign_from_checkpoint(model_path, slim.get_variables_to_restore(), ignore_missing_vars=True)
                sess.run(init_assign_op, feed_dict=init_feed_dict)
                sess.run(global_step.assign(0))
                '''
            train_handle = sess.run(train_iterator.string_handle())
            valid_handle = sess.run(valid_iterator.string_handle())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])

            # add loss dict
            imgId_loss_dict = {}
            print("Start training...")
            for step in range(total_step_num):
                start_time = time.time()
                _, loss_value, lh_loss, lr_real, imgIds_, loss_of_ids_, l_l2_, l_symm_, l_dist_ = \
                    sess.run([train_op, loss, last_heat_loss, learning_rate, imgIds, loss_of_ids, l_l2, l_symm, l_dist],
                                                  feed_dict={handle: train_handle})

                if step == 0:
                    format_str = ('loss=%.3f, l_l2=%.3f,l_sy=%.3f,l_ds=%.3f,last_heat_loss=%.3f')
                    print(format_str % (loss_value, l_l2_, l_symm_, l_dist_, lh_loss))

                # for loss dict, skip first 10 epochs
                num_of_steps_an_epoch = (params['num_train_samples'] // (params['batchsize'] * params['gpus']))
                if step > 1 * num_of_steps_an_epoch:
                    update_imgId_loss_dict(imgId_loss_dict, imgIds_, loss_of_ids_)

                if step != 0 and step % num_of_steps_an_epoch == 0:
                    # print('imgId_loss_dict==>',imgId_loss_dict)
                    print('size of imgId_loss_dict ==> ', len(imgId_loss_dict))
                    with open(imgid_loss_file + '_out.json', 'w') as f:
                        f.write(json.dumps({str(k):str(v) for k, v in imgId_loss_dict.items()}))
                        print("JSON WRITTEN !")


                duration = time.time() - start_time

                if step != 0 and step % params['per_update_tensorboard_step'] == 0:
                    # False will speed up the training time.
                    if params['pred_image_on_tensorboard'] is True:
                        valid_loss_value, valid_lh_loss, valid_in_image, valid_in_heat, valid_p_heat = sess.run(
                            [loss, last_heat_loss, input_image, input_heat, pred_heat],
                            feed_dict={handle: valid_handle}
                        )
                        result = []
                        for index in range(params['batchsize']):
                            r = CocoPose.display_image(
                                    valid_in_image[index,:,:,:],
                                    valid_in_heat[index,:,:,:],
                                    valid_p_heat[index,:,:,:],
                                    True
                                )
                            result.append(
                                r.astype(np.float32)
                            )

                        comparsion_of_pred_result = sess.run(
                            pred_result__summary,
                            feed_dict={
                                pred_result_image: np.array(result)
                            }
                        )
                        summary_writer.add_summary(comparsion_of_pred_result, step)

                    # print train info
                    num_examples_per_step = params['batchsize'] * params['gpus']
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / params['gpus']
                    format_str = ('%s: step %d, lr=%f, loss=%.3f, l_l2=%.3f,l_sy=%.3f,l_ds=%.3f,last_heat_loss=%.3f (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), step, lr_real, loss_value, l_l2_, l_symm_, l_dist_, lh_loss, examples_per_sec, sec_per_batch))

                    # tensorboard visualization
                    merge_op = sess.run(summary_merge_op, feed_dict={handle: valid_handle})
                    summary_writer.add_summary(merge_op, step)

                # save model
                if step != 0 and step % params['per_saved_model_step'] == 0:
                    checkpoint_path = os.path.join(params['modelpath'], training_name, 'model')
                    saver.save(sess, checkpoint_path, global_step=step)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
