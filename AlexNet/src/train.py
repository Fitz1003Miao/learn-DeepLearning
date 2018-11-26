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

    data_train_placeholder = tf.placeholder(data_train.dtype, data_train.shape, name = 'data_train')
    label_train_placeholder = tf.placeholder(tf.int64, label_train.shape, name = 'label_train')
    data_val_placeholder = tf.placeholder(data_val.dtype, data_val.shape, name = 'data_val')
    label_val_placeholder = tf.placeholder(tf.int64, label_val.shape, name = 'label_val')
    handle = tf.placeholder(tf.string, shape = [], name = 'handle')

    dataset_train = tf.data.Dataset.from_sparse_tensor_slices((data_train_placeholder, label_train_placeholder))

    dataset_train = data_train.repeat(num_epochs)
    iterator_train = dataset_train.make_initializable_iterator()
    batch_num_per_epoch = math.ceil(num_train / batch_size)

    dataset_val = tf.data.Dataset.from_sparse_tensor_slices((data_val_placeholder, label_val_placeholder))
    batch_num_val = math.ceil(num_val / batch_size)
    iterator_val = dataset_val.make_initializable_iterator()

    iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types)
    (datas, labels) = iterator.get_next()

    net = AlexNet(input = datas, setting = setting)
    logits = net.logits
    probs = tf.nn.softmax(logits, name = 'probs')
    predictions = tf.argmax(probs, axis = -1, name = 'predictions')
    
    labels_2d = tf.expand_dims(labels, axis = -1, name = 'label_2d')
    labels_tile = tf.tile(labels_2d, )
    
if __name__ == "__main__":
    main()