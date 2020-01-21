import tensorflow as tf
import tensorflow.contrib.slim as slim

 
_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_trainable = True


    
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

if __name__ == '__main__':
    input = tf.zeros(shape=[1,224,224,32])
    #output = conv2x2_base_block(input, 16, 'yang', add=False)
    #output = conv2x2_overall_block(output, 16, 2, 'yang')
    #output = base_block_2x2('CONV', input, 16, 'yang', add=False)
    #output = overall_block_2x2('CONV', output, 16, 2, 'fei', stride=None)
    #print(output.get_shape().as_list())
    
    use_2x2 = False
    import time
    t1 = time.time()
    N = 100
    if not use_2x2:
        for i in range(N):
            input = slim.separable_convolution2d(input,
                                               num_outputs=None,
                                               stride=1,
                                               depth_multiplier=1.0,
                                               kernel_size=5,
                                               activation_fn=None,
                                               weights_initializer=_init_xavier,
                                               weights_regularizer=_l2_regularizer_00004,
                                               biases_initializer=None,
                                               normalizer_fn=slim.batch_norm,
                                               #normalizer_fn=None,
                                               #padding="SAME",
                                               scope='dw%d'%i,
                                               trainable=_trainable,
                                               padding='SAME',
                                               )
    else:
        for i in range(N):
            input = overall_block_2x2('SEPARABLE', input, 
                         num_filters=32,
                         num_blocks=(5 * 5) // 8 - 1, 
                         name="dw%d"%i, 
                         stride=None)
    
    t2=time.time()
    print('time ==>', t2-t1)
    #tensor = tf.get_default_graph().get_tensor_by_name("dw0/depthwise:0")
    tensor = tf.get_default_graph().get_tensor_by_name("dw0/depthwise_weights:0")
    #tensor = tf.get_default_graph().get_tensor_by_name("dw0/0/SEPARABLE2x2_0/depthwise_weights:0")
    #tensor = tf.get_default_graph().as_graph_def().node
    #print(tensor)
    import numpy as np
    a = np.random.random([1,224,224,32])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        w = sess.run(tensor, feed_dict={input: a})
    #print(w, w.shape)


