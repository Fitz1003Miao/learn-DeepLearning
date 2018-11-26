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

    data_train, label_train = utils.load_data(train_file)
    data_val, label_val = utils.load_data(val_file)

    data_train, label_train = utils.shuffle(data_train, label_train)
    num_train = data_train.shape[0]
    num_val = data_val.shape[0]

    # 实现细节：
    # 1. 采用 batch size 为 128 的 SGD
    # 2. 采用 momentum 为 0.9 的 MomentumOptimizer
    # 3. 采用 weight_decay 为 0.0005 的 regularzation loss
    # 4. 初始化每一层 weights 均值为0，标准差为 0.01
    # 5. 初始化第二、四、五卷积层以及 所有的全连接层 的 bias 为 1
    # 这么做的好处是能够通过 Relu 对学习速率提升
    # 6. 运行 90 个循环

    # 定义占位符
    data_train_placeholder = tf.placeholder(data_train.dtype, shape = (None, 3), name = 'data_train_placeholder')
    label_train_placeholder = tf.placeholder(tf.int32, shape = (None,), name = 'label_train_placeholder')
    data_val_placeholder = tf.placeholder(data_val.dtype, shape = (None, 3), name = 'data_train_placeholder')
    label_val_placeholder = tf.placeholder(tf.int32, shape = (None,), name = 'label_val_placeholder')

    # 计算每次epoch所需要的batch 数量
    batch_num_per_epoch = math.ceil(num_train / batch_size)
    
    # 定义网络结构并取得预测结果
    net = AlexNet(input = data_train_placeholder, setting = setting)
    logits = net.logits
    
    # 计算loss，分为两部分，一部分是预测loss，另外一部分是 正则化loss
    labels_2d = tf.expand_dims(labels, axis = -1, name = 'labels_2d')
    labels_tile = tf.tile(labels_2d, (1, tf.shape(logits)[-1]), name = 'labels_tile')
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels = labels_tile, logits = logits)

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    optimizer = tf.train.MomentumOptimizer(learning_rate = setting.learning_rate_base, momentum = setting.momentum, use_nesterov = True)
    train_op = optimizer.minimize(loss_op + reg_loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(num_epochs):
            for batch_idx in range(batch_num_per_epoch):
                sess.run(train_op, feed_dict = {data_train_placeholder : data_train[batch_idx * batch_size : min(num_train, (batch_idx + 1) * batch_size)], label_train_placeholder : label_train[batch_idx * batch_size : min(num_train, (batch_idx + 1) * batch_size)]})
            if epoch != 0 and epoch % 5 == 0:
                sess.run(loss_op + reg_loss, feed_dict = {data_train_placeholder : data_train, label_train_placeholder : label_train})

if __name__ == "__main__":
    main()