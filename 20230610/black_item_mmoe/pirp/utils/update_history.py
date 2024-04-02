# coding:utf-8

import tensorflow as tf
import os


class UpdateHistory(tf.keras.callbacks.Callback):

    def __init__(self, log_file, ds=None):
        super().__init__()
        # log file
        self.log_file = log_file
        # train end ds
        self.ds = ds

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        with open(self.log_file, 'a+') as f:
            f.write('[{0}][Epoch = {1} End] log = {2}\n'.format(self.ds, epoch, logs))

    def on_train_end(self, logs=None):
        with open(self.log_file, 'a+') as f:
            f.write('[{0}][Train End] log = {1}\n'.format(self.ds, logs))

    def on_test_end(self, logs=None):
        # called at the end of evaluation or validation
        with open(self.log_file, 'a+') as f:
            f.write('[Test End] log = {0}\n\n'.format(logs))

