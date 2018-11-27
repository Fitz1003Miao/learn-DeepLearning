#-*- coding: utf-8 -*-
import tensorflow as tf
import argparse
from AlexNet import AlexNet
import importlib
import utils
import math
import os
import shutil
from datetime import datetime
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("-t", "--train", help = "Path to train file")
    parse.add_argument("-v", "--val", help = "Path to val file")
    parse.add_argument("-s", "--save_folder", help = "Path to save")
    parse.add_argument("-x", "--setting", help = "setting to import")

    args = parse.parse_args()
    print(args)

    train_file = args.train
    val_file = args.val

    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    root_folder = os.path.join(args.save_folder, '%s_%s_%d' % (args.setting, time_string, os.getpid()))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder, exist_ok = True)

    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = setting.batch_size
    step_val = setting.step_val

    print('loading data...')
    data_train, label_train = utils.load_data(train_file)
    data_val, label_val = utils.load_data(val_file)

    data_train, label_train = utils.shuffle(data_train, label_train)
    num_train = data_train.shape[0]
    num_val = data_val.shape[0]

    print('load {} train data, {} val data'.format(num_train, num_val))

    # 实现细节：
    # 1. 采用 batch size 为 128 的 SGD
    # 2. 采用 momentum 为 0.9 的 MomentumOptimizer
    # 3. 采用 weight_decay 为 0.0005 的 regularzation loss
    # 4. 初始化每一层 weights 均值为0，标准差为 0.01
    # 5. 初始化第二、四、五卷积层以及 所有的全连接层 的 bias 为 1
    # 这么做的好处是能够通过 Relu 对学习速率提升
    # 6. 运行 90 个循环

    # 定义占位符
    data_train_placeholder = tf.placeholder(data_train.dtype, shape = [None, 224, 224, 3], name = 'data_train_placeholder')
    label_train_placeholder = tf.placeholder(tf.int32, shape = [None], name = 'label_train_placeholder')
    data_val_placeholder = tf.placeholder(data_val.dtype, shape = [None, 224, 224, 3], name = 'data_train_placeholder')
    label_val_placeholder = tf.placeholder(tf.int32, shape = [None], name = 'label_val_placeholder')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    handle = tf.placeholder(tf.string, shape=[], name='handle')

    dataset_train = tf.data.Dataset.from_tensor_slices((data_train_placeholder, label_train_placeholder))

    dataset_train = dataset_train.shuffle(buffer_size = 4 * batch_size)
    dataset_train = dataset_train.repeat(count = num_epochs)

    dataset_train = dataset_train.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
    dataset_val = dataset_val.repeat(count = num_epochs)
    dataset_val = dataset_val.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    iterators = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
    (data, label) = iterators.get_next()
    
    train_val_iterators = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    iterators_train = train_val_iterators.make_initializer(dataset_train)
    iterators_val = train_val_iterators.make_initializer(dataset_val)
    # data.set_shape([batch_size, 224, 224, 3])
    # label.set_shape([batch_size, ])
    
    # 计算每次epoch所需要的batch 数量
    batch_num_per_epoch = math.ceil(num_train / batch_size)
    batch_num = batch_num_per_epoch * num_epochs
    batch_num_val = math.ceil(num_val / batch_size)
    
    # 定义网络结构并取得预测结果
    net = AlexNet(input = data, setting = setting)
    logits = net.logits
    
    probs = tf.nn.softmax(logits, name = 'probs')
    predictions = tf.argmax(probs, axis = -1, name = 'predictions')

    # 计算loss，分为两部分，一部分是预测loss，另外一部分是 正则化loss
    # labels_2d = tf.expand_dims(label_train_placeholder, axis = -1, name = 'labels_2d')
    # labels_tile = tf.tile(labels_2d, (1, tf.shape(logits)[1]), name = 'labels_tile')
    labels = tf.one_hot(label, setting.CLASS_NUM)
    loss_op = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)

    with tf.name_scope('metrics'):
        loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
        t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(label, predictions)
        t_1_per_class_acc_op, t_1_per_class_acc_update_op = tf.metrics.mean_per_class_accuracy(label, predictions, setting.CLASS_NUM)

    reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables() if var.name.split('/')[0] == 'metrics'])

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    lr_exp_op = tf.train.exponential_decay(learning_rate = setting.learning_rate_base, global_step = global_step, decay_rate = setting.decay_rate, decay_steps = setting.decay_steps, staircase = True)
    lr_clip_op = tf.maximum(setting.learning_rate_min, lr_exp_op)

    optimizer = tf.train.MomentumOptimizer(learning_rate = lr_clip_op, momentum = setting.momentum, use_nesterov = True)
    train_op = optimizer.minimize(loss_op + reg_loss, global_step = global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        handle_train = sess.run(train_val_iterators.string_handle())
        handle_val = sess.run(train_val_iterators.string_handle())
        sess.run(iterators_train, feed_dict = {data_train_placeholder:data_train, label_train_placeholder:label_train})

        for batch_idx_train in range(batch_num):
            if (batch_idx_train % setting.step_val == 0 and batch_idx_train != 0) or batch_idx_train == batch_num - 1:
                
                sess.run(iterators_val, feed_dict = {data_val_placeholder : data_val, label_val_placeholder : label_val})
                sess.run(reset_metrics_op)

                for batch_idx_val in range(batch_num_val):
                    sess.run([loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op], feed_dict = {handle : handle_val})

                loss_val, t_1_acc_val, t_1_per_class_acc_val = sess.run([loss_mean_op, t_1_acc_op, t_1_per_class_acc_op])
                print('Average:  Loss:{:.4f}  T-1 Acc:{:.4f}  T-1 mAcc:{:.4f}'.format(loss_val, t_1_acc_val, t_1_per_class_acc_val))

            sess.run(reset_metrics_op)
            sess.run([train_op, loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op], feed_dict = {handle : handle_train})
            if batch_idx_train % 10 == 0:
                loss, t_1_acc, t_1_per_class_acc = sess.run([loss_mean_op, t_1_acc_op, t_1_per_class_acc_op])
                print('[Train] - Iter : {:06d}    Loss:{:.4f}   T-1 Acc:{:.4f}   T-1 mAcc:{:.4f}'.format(batch_idx_train, loss, t_1_acc, t_1_per_class_acc))
if __name__ == "__main__":
    main()