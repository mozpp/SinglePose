import tensorflow as tf
import numpy as np

# tf.enable_eager_execution()
print("Eager execution mode: ", tf.executing_eagerly())

parts = ["top_head", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
         "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]
point_conterparts = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)]
line_conterparts = [((1, 2), (1, 5)),
                    ((2, 3), (5, 6)),
                    ((3, 4), (6, 7)),
                    ((1, 8), (1, 11)),
                    ((8, 9), (11, 12)),
                    ((9, 10), (12, 13)),]


def focal_loss_sigmoid(y_pred, y_true, alpha = 2.0, beta=4):
    '''
    this function return the focal loss between gaussian heatmap
    and prediction.
    if the input shape include batch size, for example [n, 112, 112, 14],
    the calculated result is loss per batch
    '''
    epsilon = 1.e-7

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    max = tf.reduce_max(y_true, axis=(1, 2))
    max = tf.where(tf.equal(max, 0), -1*tf.ones_like(max), max)
    max = tf.stack([max] * y_true.get_shape().as_list()[1], axis=1)
    max = tf.stack([max] * y_true.get_shape().as_list()[2], axis=2)

    y_true_mask = tf.where(tf.equal(y_true, max), tf.ones_like(max), tf.zeros_like(max))
    y_true_mask = tf.cast(y_true_mask, tf.float32)


    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    # y_pred = tf.sigmoid(y_pred)

    gauss = tf.pow(1 - y_true, beta)
    y_t = tf.multiply(y_pred, y_true_mask) + tf.multiply(1-y_pred, 1-y_true_mask) #* gauss
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), alpha)
    fl = tf.multiply(weight, ce)

    loss = tf.reduce_mean(fl)
    return loss

def custom_l2_loss(y_pred, y_true):
    loss = 0.5 * tf.reduce_sum(tf.pow(y_true - y_pred, 2), axis=(1,2,3))
    return loss


def softargmax(x, axis, beta=1e0):
    '''
    this fun return the softmax argmax
    :param x: tensor of abitary shape
    :param beta: 1 for regular softmax and inf for argmax
    :return: [a, b] when x is [a, b, c], aixs=1,
    '''
    x = tf.convert_to_tensor(x)
    x_range = tf.range(x.shape.as_list()[axis], dtype=x.dtype)
    for i in range(len(x.shape.as_list())):
        if i != axis:
            x_range = tf.expand_dims(x_range, i)
    return tf.reduce_sum(tf.nn.softmax(x*beta, axis=axis) * x_range, axis=axis)


def get_max_location(tensor):
    '''
    Get max location of 4D tensor
    :param tensor: [n, h, w, c]
    :return: [n, c, 2]
    '''
    assert len(tensor.get_shape().as_list()) == 4

    x_maxs = tf.reduce_max(tensor, axis=1)
    y_maxs = tf.reduce_max(tensor, axis=2)

    # x = tf.argmax(y_maxs, axis=1)
    # y = tf.argmax(x_maxs, axis=1)

    # use softargmax to get weighted locations and non-zero gradients
    x = softargmax(y_maxs, axis=1)
    y = softargmax(x_maxs, axis=1)

    loc = tf.stack([x, y], axis=-1)
    return loc

def calc_distance(p1, p2):
    '''
    :param p1: tensor [n, 2]
    :param p2: tensor [n, 2]
    :return: tensor [n,]
    '''

    d = tf.reduce_sum(tf.pow(p1-p2, 2), axis=-1)
    d = tf.cast(d, tf.float32)
    d = tf.sqrt(d)
    return d


def distance_loss(heat_maps):
    '''
    This loss penalizes the deviation in same type of limbs
    :param heat_maps: [n, 112, 112, 14] shaped tensor
    :return: scalar, the loss
    '''
    # convert type
    if not heat_maps.dtype is tf.float32:
        heat_maps = tf.cast(heat_maps, tf.float32)

    # locs shape: [n, c, 2]
    locs = get_max_location(heat_maps)

    losses = []
    for pair in line_conterparts:
        l1, l2 = pair
        c11, c12 = l1
        c21, c22 = l2

        # [n, 2]
        p11 = locs[:, c11]
        p12 = locs[:, c12]
        p21 = locs[:, c21]
        p22 = locs[:, c22]

        def calc_d(p1, p2):
            '''
            filt (0 ,0) points
            :param p1: [n, 2]
            :param p2: [n, 2]
            :return: distances, 0 when (0,0) is included in calc [n,]
            '''
            def get_mask(p):
                zeros = tf.equal(p, tf.constant([[0, 0]], tf.float32))
                p_mask = tf.reduce_all(zeros, axis=-1)
                return p_mask

            p1_mask = get_mask(p1)
            p2_mask = get_mask(p2)
            p_mask = tf.logical_or(p1_mask, p2_mask)

            d = calc_distance(p1, p2)
            d = d * tf.cast(tf.logical_not(p_mask), tf.float32)
            return tf.cast(d, tf.float32)

        d1 = calc_d(p11, p12)
        d2 = calc_d(p21, p22)
        d_mean = (d1 + d2) / tf.constant(2.)

        # relative distance
        diff = tf.abs(d1 - d2) / (d_mean + tf.constant(1e-9))
        # enlarge penality
        diff = tf.pow(diff, 4)
        losses.append(diff)
    loss = tf.reduce_mean(losses, axis=0)
    return loss

def symmetrical_loss(heat_maps, feature_maps):
    '''
    This loss restricts the symmetry properties of a person
    :param heat_maps: [n, 112, 112, 14] shaped tensor
    :param feature_maps: list of [n, x, x, y] shaped tensors
    :return: list of scalars, one for each feature map
    '''

    # convert type
    if not heat_maps.dtype is tf.float32:
        heat_maps = tf.cast(heat_maps, tf.float32)
    if not feature_maps[0].dtype is tf.float32:
        feature_maps = [tf.cast(fm, tf.float32) for fm in feature_maps]

    # generate heat masks
    # heat_mask = tf.zeros_like(heat_maps)
    # heat_mask = tf.where(tf.equal(heat_maps, 0), heat_mask, tf.ones_like(heat_maps))
    heat_mask = heat_maps

    # resize feature maps
    h, w = heat_maps.get_shape().as_list()[1:3]
    channels = feature_maps[0].get_shape().as_list()[-1]

    resized_fms = [tf.image.resize(fm, (h, w)) for fm in feature_maps]

    # get features and compare
    losses = []
    for fm in resized_fms:
        losses_of_fm = []
        for pair in point_conterparts:
            c1, c2 = pair
            mask1 = tf.stack([heat_mask[:,:,:,c1]] * channels, axis=-1)
            mask2 = tf.stack([heat_mask[:,:,:,c2]] * channels, axis=-1)
            # plot_tensor(mask1)
            # plot_tensor(fm)

            feature1 = fm * mask1
            feature2 = fm * mask2
            # plot_tensor(feature1)

            loss = custom_l2_loss(feature1, feature2)
            losses_of_fm.append(loss)
        scalar_loss_of_fm = tf.reduce_mean(losses_of_fm, axis=0)
        losses.append(scalar_loss_of_fm)
    return losses

def plot_tensor(tensor):
    '''
    only used when eager execution is enabled
    '''
    import matplotlib.pyplot as plt

    print("Shape of Tensor: ", tensor.get_shape().as_list())
    n, h, w, c = tensor.get_shape().as_list()
    if n==None:
        return
    fig, axs = plt.subplots(n, c)
    axs = np.reshape(axs, [-1])
    for i in range(n * c):
        axs[i].imshow(tensor[i//c, :, :, i%c], plt.cm.gray)
    plt.savefig('tmp_tensor.jpg')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    test_symm_loss = False
    test_distance_loss = True

    if test_symm_loss:
        a = np.zeros([2, 112, 112, 14])
        a[:, 10:20, 30:40, :] = 0.3
        a[...,2] = 0

        b = np.zeros([2,28,28,3])
        b[:, :15, :20, :] = 0.5

        c = np.zeros([2,56,56,3])
        c[:, 10:15, 15:20, :] = 0.9

        heat_maps = tf.Variable(shape=[2, 112, 112, 14], initial_value=a)
        feature_maps = [tf.Variable(shape=[2, 28, 28, 3], initial_value=b),
                        tf.Variable(shape=[2, 56, 56, 3], initial_value=c)]
        symm_losses = symmetrical_loss(heat_maps, feature_maps)



        ''' eager mode '''
        # print(symm_losses)

        ''' graph mode '''
        # plot_tensor(heat_maps)
        # for tensor in feature_maps:
        #     plot_tensor(tensor)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            symm_losses_ = sess.run(symm_losses)
            for loss in symm_losses_:
                print(loss)

    if test_distance_loss:
        # np.random.seed(1)
        # a = np.random.randint(0, 10, [2,3,4,14])
        # print(a)

        # # test calc_distance
        # p1 = tf.constant(np.random.randint(0, 10, [3,2]))
        # p2 = tf.constant(np.random.randint(0, 10, [3,2]))
        # d = calc_distance(p1, p2)
        # print(p1,p2,d)

        # a = tf.constant(a)
        # a = tf.Variable(a, trainable=True, name="a")
        # loss = distance_loss(a)



        opt = tf.train.AdamOptimizer(0.001)

        np.random.seed(1)
        x = np.random.randint(0, 10, [1, 3, 4, 14])
        print('input==>',x)

        a = tf.Variable(x, trainable=True, name="input", dtype=tf.float32)
        loss = distance_loss(a)

        grad = opt.compute_gradients(loss)



        print(grad)

        ''' eager test '''
        # locs = get_max_location(a)
        # print('==>, locs)
        # print('==>', loss)

        ''' graph test '''
        # with tf.Session() as sess:
        #     loss_ = sess.run(loss)
        #     print(loss_)


