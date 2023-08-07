# coding:utf-8

import tensorflow as tf


class Bucketize(tf.keras.layers.Layer):
    def __init__(self, boundaries, **kwargs):
        self.boundaries = boundaries
        super(Bucketize, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Bucketize, self).build(input_shape)

    def call(self, x, **kwargs):
        return tf.raw_ops.Bucketize(input=x, boundaries=self.boundaries)

    def get_config(self,):
        config = {'boundaries': self.boundaries}
        base_config = super(Bucketize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
