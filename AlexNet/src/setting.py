CLASS_NUM = 2

batch_size = 128
num_epochs = 90
step_val = 5
label_weights = [1.0] * CLASS_NUM

learning_rate_base = 0.01
learning_rate_min = 0.0001
momentum = 0.9
weight_decay = 0.0005

decay_steps = 5000
decay_rate = 0.8

convLayer_param_name = ('kernel_num', 'conv_kernel_height', 'conv_kernel_width', 'conv_strideX', 'conv_strideY', 'conv_name', 'bias', 'group', 'conv_padding', 'lrn_index', 'pool_index')
convLayer_params = [dict(zip(convLayer_param_name, convLayers_param)) for convLayers_param in 
                     [(96, 11, 11, 4, 4, "conv1", 1, 1, "SAME", 0, 0),
                      (256, 5, 5, 1, 1, "conv2", 1, 2, "SAME", 1, 1),
                      (384, 3, 3, 1, 1, "conv3", 1, 1, "SAME", None, None),
                      (384, 3, 3, 1, 1, "conv4", 1, 2, "SAME", None, None),
                      (256, 3, 3, 1, 1, "conv5", 1, 2, "SAME", None, 2)]]

lrn_param_name = ('lrn_k', 'lrn_bias', 'lrn_alpha', 'lrn_beta', 'lrn_name')
lrn_params = [dict(zip(lrn_param_name, lrn_param)) for lrn_param in 
                [(2, 1, 1e-04, 0.75, "norm1"),
                 (2, 1, 1e-04, 0.75, "norm2")]]

pool_param_name = ('pool_kernel_height', 'pool_kernel_width', 'pool_strideX', 'pool_strideY', 'pool_name', 'pool_padding')
pool_params = [dict(zip(pool_param_name, pool_param)) for pool_param in 
                [(3, 3, 2, 2, "pool1", "VALID"),
                 (3, 3, 2, 2, "pool2", "VALID"),
                 (3, 3, 2, 2, "pool5", "VALID")]]

fcLayer_param_name = ('inputD', 'outputD', 'KeepPro', 'name')
fcLayer_params = [dict(zip(fcLayer_param_name, fcLayers_param)) for fcLayers_param in 
                    [(256 * 6 * 6, 4096, 0.5, 'fc1'), 
                     (4096, 4096, 0.5, 'fc2'), 
                     (4096, CLASS_NUM, None, 'fc3')]]