import tensorflow as tf

def maxPoolLayer(input, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    return tf.nn.max_pool(input, ksize = [1, kHeight, kWidth, 1], strides = [1, strideX, strideY, 1], padding = padding, name = name)

def LRN(input, k, bias, alpha, beta, name):
    return tf.nn.local_response_normalization(input = input, depth_radius = k,bias = bias, alpha = alpha, beta = beta, name = name)

def dropout(input, keepPro, name = None):
    return tf.nn.dropout(input, keep_prob = keepPro, name = name)

def convLayer(input, filter_num, kHeight, kWidth, strideX, strideY, name, group = 1, padding = "SAME"):
    channel = int(input.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('k', shape = [kHeight, kWidth, channel / group, filter_num])
        b = tf.get_variable('b', shape = [filter_num])
        inputs = tf.split(axis = -1, value = input, num_or_size_splits = group)
        kernels = tf.split(axis = -1, value = kernel, num_or_size_splits = group)
        bs = tf.split(axis = -1, value = b, num_or_size_splits = group)

        res = []
        for (x, k, b) in zip(inputs, kernels, bs):
            res.append(tf.nn.bias_add(tf.nn.conv2d(x, k, strides = [1, strideX, strideY, 1], padding = padding), b))
        return tf.concat(res, axis = -1)
        
def fcLayer(input, inputD, outputD, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape = [inputD, outputD], dtype = 'float')
        b = tf.get_variable('b', shape = [outputD], dtype = 'float')

        out = tf.nn.xw_plus_b(input, w, b, name = scope.name)

        return tf.nn.relu(out, name = scope.name)

class AlexNet(object):
    def __init__(self, input, setting):
        output = []
        output.append(input)
        for convLayer_param in setting.convLayer_params:

            conv_kernel_num = convLayer_param['kernel_num']
            conv_kernel_Height = convLayer_param['conv_kernel_height']
            conv_kernel_width = convLayer_param['conv_kernel_width']
            conv_kernel_strideX = convLayer_param['conv_strideX']
            conv_kernel_strideY = convLayer_param['conv_strideY']
            conv_name = convLayer_param['conv_name']
            group = convLayer_param['group']
            conv_padding = convLayer_param['conv_padding']
            lrn_index = convLayer_param['lrn_index']
            pool_index = convLayer_param['pool_index']

            input = output[-1]

            out = convLayer(input = input, filter_num = conv_kernel_num, kHeight = conv_kernel_Height, kWidth = conv_kernel_width, strideX = conv_kernel_strideX, strideY = conv_kernel_strideY, name = conv_name, group = group, padding = conv_padding)
            
            if lrn_index is not None:
                lrn_k = setting.lrn_params[lrn_index]['lrn_k']
                lrn_bias = setting.lrn_params[lrn_index]['lrn_bias']
                lrn_alpha = setting.lrn_params[lrn_index]['lrn_alpha']
                lrn_beta = setting.lrn_params[lrn_index]['lrn_beta']
                lrn_name = setting.lrn_params[lrn_index]['lrn_name']

                out = LRN(input = out, k = lrn_k, bias = lrn_bias, alpha = lrn_alpha, beta = lrn_beta, name = lrn_name)

            if pool_index is not None:
                pool_kernel_height = setting.pool_params[pool_index]['pool_kernel_height']
                pool_kernel_width = setting.pool_params[pool_index]['pool_kernel_width']
                pool_strideX = setting.pool_params[pool_index]['pool_strideX']
                pool_strideY = setting.pool_params[pool_index]['pool_strideY']
                pool_name = setting.pool_params[pool_index]['pool_name']
                pool_padding = setting.pool_params[pool_index]['pool_padding']

                out = maxPoolLayer(input = out, kHeight = pool_kernel_height, kWidth = pool_kernel_width, strideX = pool_strideX, strideY = pool_strideY, name = pool_name, padding = pool_padding)
            output.append(out)
        output[-1] = tf.reshape(output[-1], [-1, 256 * 6 * 6])
        for fcLayer_param in setting.fcLayer_params:
            fcIn = output[-1]

            inputD = fcLayer_param['inputD']
            outputD = fcLayer_param['outputD']
            keepPro = fcLayer_param['KeepPro']
            name = fcLayer_param['name']

            fc = fcLayer(input = fcIn, inputD = inputD, outputD = outputD, name = name)
            if keepPro is not None:
                out = dropout(fc, keepPro)
            else:
                out = fc
            output.append(out)

        self.logit = output[-1]
        

