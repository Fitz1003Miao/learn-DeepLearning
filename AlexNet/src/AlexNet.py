import tensorflow as tf

def maxPoolLayer(intput, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(input = intput, ksize = [1, kHeight, kWidth, 1], strides = [1, strideX, strideY, 1], padding = padding, name = name)

def LRN(intput, k, bias, alpha, beta, name):
    return tf.nn.local_response_normalization(input = intput, depth_radius = k,bias = bias, alpha = alpha, beta = beta, name = name)

def dropout(intput, keepPro, name = None):
    return tf.nn.dropout(intput, keep_prob = keepPro, name = name)

def conv(intput, filter_num, kHeight, kWidth, strideX, strideY, name, group = 1, padding = "SAME"):
    with tf.variable_scope(name) as scope:
        res = tf.nn.conv2d(input = intput, filter = [kHeight, kWidth, intput.shape[-1], filter_num], strides = [1, strideX, strideY, 1], padding = padding)
        resNew = tf.split(value = res, num_or_size_splits = group, axis = -1)
        reslist = [tf.nn.relu(x, name = scope.name) for x in resNew]
        return tf.concat(concat_dim = -1, values = reslist, name = scope.name)

def fcLayer(intput, inputD, outputD, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape = [inputD, outputD], dtype = 'float')
        b = tf.get_variable('b', shape = [outputD], dtype = 'float')

        out = tf.nn.xw_plus_b(intput, w, b, name = scope.name)

        return tf.nn.relu(out, name = scope.name)

class AlexNet(object):
    def __init__(self, x):
        conv1 = conv(intput = x, filter_num = 96, kHeight = 11, kWidth = 11, strideX = 4, strideY = 4, name = "conv1", group = 1, padding = "SAME")
        lrn1 = LRN(intput = conv1, k = 2, alpha = 2e-05, beta = 0.75, name = "norm1")
        pool1 = maxPoolLayer(input = lrn1, kHeight = 3, kWidth = 3, strideX = 2, strideY = 2, name = "pool1", padding = "SAME")

        conv2 = conv(intput = pool1, filter_num = 256, kHeight = 5, kWidth = 5, strideX = 1, strideY = 1, name = "conv2", group = 2)
        lrn2 = LRN(intput = conv2, k = 2, alpha = 2e-05, beta = 0.75, name = "norm2")
        pool2 = maxPoolLayer(intput = lrn2, kHeight = 3, kWidth = 3, strideX = 2, strideY = 2, name = "pool2")

        conv3 = conv(intput = pool2, filter_num = 384, kHeight = 3, kWidth = 3, strideX = 1, strideY = 1, name = "conv3")

        conv4 = conv(intput = conv3, filter_num = 384, kHeight = 3, kWidth = 3, strideX = 1, strideY = 1, name = "conv4", group = 2)

        conv5 = conv(intput = conv4, filter_num = 256, kHeight = 3, kWidth = 3, strideX = 1, strideY = 1, name = "conv5", group = 2)
        pool5 = maxPoolLayer(intput = conv5, kHeight = 3, kWidth = 3, strideX = 2, strideY = 2, name = "pool5", padding = "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])