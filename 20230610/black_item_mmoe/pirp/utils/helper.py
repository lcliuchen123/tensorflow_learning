# coding: utf-8

import tensorflow as tf


def get_dtype(data_type):
    if data_type == 'int':
        return tf.int32
    elif data_type == 'long':
        return tf.int64
    elif data_type == 'float':
        return tf.float32
    else:
        return tf.string
