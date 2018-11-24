import tensorflow as tf

def maxPoolLayer(input, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(input = input, ksize = [1, kHeight, kWidth, 1], strides = [1, strideX, strideY, 1], padding = padding, name = name)

def LRN(input, k, bias, alpha, beta, name):
    return tf.nn.local_response_normalization(input = input, depth_radius = k,bias = bias, alpha = alpha, beta = beta, name = name)

def dropout(input, keepPro, name = None):
    return tf.nn.dropout(input, keep_prob = keepPro, name = name)

def conv(input, filter_num, kHeight, kWidth, strideX, strideY, name, group = 1, padding = "SAME"):
    channel = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        kernel = tf.get_varibale('k', shape = [kHeight, kWidth, channel / group, filter_num])
        
        inputs = tf.split(axis = -1, value = input, num_or_size_splits = group)
        kernels = tf.split(axis = -1, value = kernel, num_or_size_splits = group)

        res = []
        for (x, k) in zip(inputs, kernels):
            res.append(tf.nn.conv2d(x, k, strides = [1, strideX, strideY, 1], padding = padding))
        return tf.concat(res, axis = -1)
        
def fcLayer(input, inputD, outputD, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape = [inputD, outputD], dtype = 'float')
        b = tf.get_variable('b', shape = [outputD], dtype = 'float')

        out = tf.nn.xw_plus_b(input, w, b, name = scope.name)

        return tf.nn.relu(out, name = scope.name)

class AlexNet(object):
    def __init__(self, input, setting):
        self.out = []
        conv1 = conv(input = input, filter_num = 96, kHeight = 11, kWidth = 11, strideX = 4, strideY = 4, name = "conv1", group = 1, padding = "SAME")
        lrn1 = LRN(input = conv1, k = 2, bias = 1, alpha = 2e-05, beta = 0.75, name = "norm1")
        pool1 = maxPoolLayer(input = lrn1, kHeight = 3, kWidth = 3, strideX = 2, strideY = 2, name = "pool1", padding = "VALID")

        conv2 = conv(input = pool1, filter_num = 256, kHeight = 5, kWidth = 5, strideX = 1, strideY = 1, name = "conv2", group = 2, padding = "SAME")
        lrn2 = LRN(input = conv2, k = 2, bias = 1, alpha = 2e-05, beta = 0.75, name = "norm2")
        pool2 = maxPoolLayer(input = lrn2, kHeight = 3, kWidth = 3, strideX = 2, strideY = 2, name = "pool2", padding = "VALID")

        conv3 = conv(input = pool2, filter_num = 384, kHeight = 3, kWidth = 3, strideX = 1, strideY = 1, name = "conv3", group = 1, padding = "SAME")

        conv4 = conv(input = conv3, filter_num = 384, kHeight = 3, kWidth = 3, strideX = 1, strideY = 1, name = "conv4", group = 2, padding = "SAME")
        
        conv5 = conv(input = conv4, filter_num = 256, kHeight = 3, kWidth = 3, strideX = 1, strideY = 1, name = "conv5", group = 2, padding = "SAME")
        pool5 = maxPoolLayer(input = conv5, kHeight = 3, kWidth = 3, strideX = 2, strideY = 2, name = "pool5", padding = "VALID")

        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        for fcLayers_param in setting.fcLayers_params:
            fcIn = self.out[-1]

            inputD = fcLayers_param['inputD']
            outputD = fcLayers_param['outputD']
            keepPro = fcLayers_param['keepPro']
            name = fcLayers_param['name']

            fc = fcLayer(input = fcIn, inputD = inputD, outputD = outputD, name = name)
            if keepPro is not None:
                out = dropout(fc, keepPro)
            else:
                out = fc
            self.out.append(out)

        fc2 = fcLayer(input = dropout1, inputD = 4096, outputD = 4096, name = "fc2")
        dropout2 = dropout(fc2, 0.5)

        self.fc3 = fcLayer(input = dropout2, inputD = 4096, outputD = setting.CLASS_NUM, name = "fc3")

