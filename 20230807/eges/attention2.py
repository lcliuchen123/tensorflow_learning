# coding:utf-8

import tensorflow as tf
from tensorflow.python.keras.regularizers import l2


class Attention_Eges(tf.keras.layers.Layer):
    def __init__(self, item_nums, l2_reg, seed, initializer, **kwargs):
        super(Attention_Eges, self).__init__(**kwargs)
        self.item_nums = item_nums
        self.seed = seed
        self.l2_reg = l2_reg
        if not initializer:
            self.initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=self.seed)
        else:
            self.initializer = initializer

    def build(self, input_shape):
        super(Attention_Eges, self).build(input_shape)
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("Attention_Eges must have two inputs")
        shape_set = input_shape
        feat_nums = shape_set[1][1]
        print("self.feat_nums: ", feat_nums)
        self.alpha_attention = self.add_weight(
            name='alpha_attention',
            shape=(self.item_nums, feat_nums),
            initializer=self.initializer,
            regularizer=l2(self.l2_reg))

    def call(self, inputs, **kwargs):
        item_input = inputs[0]
        # (batch_size, feat_nums, embed_size)
        stack_embeds = inputs[1]
        # (batch_size, 1, feat_nums)
        # alpha_embeds = tf.nn.embedding_lookup(self.alpha_attention, item_input)
        alpha_embeds = tf.gather(self.alpha_attention, item_input, axis=0)
        alpha_embeds = tf.math.exp(alpha_embeds)
        alpha_sum = tf.reduce_sum(alpha_embeds, axis=-1)
        # (batch_size, 1, embed_size)
        merge_embeds = tf.matmul(alpha_embeds, stack_embeds)
        # (batch_size, embed_size), 归一化
        merge_embeds = tf.squeeze(merge_embeds, axis=1) / alpha_sum
        return merge_embeds

    def compute_mask(self, inputs, mask):
        return None

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1][2])

    def get_config(self):
        config = {'item_nums': self.item_nums, "l2_reg": self.l2_reg, 'seed': self.seed}
        base_config = super(Attention_Eges, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
