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

    batch_num = math.ceil(num_train * num_epochs / batch_size)
    batch_num_val = math.ceil(num_val / batch_size)

    data = tf.placeholder(tf.float32, shape = (None, 224, 224, 3), name = "data")
    label = tf.placeholder(tf.int64, shape = (None, ), name = "label")
    labels_weights = tf.placeholder(tf.float32, shape=(None, ), name='labels_weights')
    global_step = tf.Variable(0, trainable = False, name='global_step')

    net = AlexNet(input = data, setting = setting)
    logits = net.logit
    probs = tf.nn.softmax(logits, name = "probs")
    predictions = tf.argmax(probs, axis = -1, name = "prdictions")

    loss_op = tf.losses.sparse_softmax_cross_entropy(labels = label, logits = logits, weights = labels_weights)
    with tf.name_scope('metrics'):
        loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
        t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(label, predictions, weights = labels_weights)
        t_1_per_class_acc_op, t_1_per_class_acc_update_op = tf.metrics.mean_per_class_accuracy(label, predictions, setting.CLASS_NUM, weights = labels_weights)

    reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables() if var.name.split('/')[0] == 'metrics'])
    
    _ = tf.summary.scalar('loss/train', tensor=loss_mean_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])
    _ = tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

    _ = tf.summary.scalar('loss/val', tensor=loss_mean_op, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_op, collections=['val'])
    _ = tf.summary.scalar('t_1_per_class_acc/val', tensor=t_1_per_class_acc_op, collections=['val'])

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps, setting.decay_rate, staircase = True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)

    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    optimizer = tf.train.MomentumOptimizer(learning_rate = lr_clip_op, momentum = setting.momentum, use_nesterov = True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep = None)

    # backup all code
    code_folder = os.path.abspath(os.path.dirname(__file__))
    shutil.copytree(code_folder, os.path.join(root_folder, os.path.basename(code_folder)))

    folder_ckpt = os.path.join(root_folder, "ckpts")
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt, exist_ok = True)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary, exist_ok = True)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])

    print(parameter_num)

if __name__ == "__main__":
    main()