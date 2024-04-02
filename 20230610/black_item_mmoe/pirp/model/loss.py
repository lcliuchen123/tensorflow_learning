# coding:utf-8

import tensorflow as tf


class OrdinalCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, target_config, name='ordinal_cross_entropy', epsilon=1e-7):
        super().__init__(name=name)
        # 目标各类分段点
        self.num_classes = target_config['bin_cnt']

        # 目标各类负样本权重
        self.neg_sample_weights = target_config['neg_sample_weights']

        # 目标各类权重
        self.class_importances = target_config['class_importances']

        # 计算精度使用
        self.epsilon = epsilon

    def label_to_one_hot(self, label):
        # 将单一label转为one-hot
        return tf.cast(tf.greater_equal(label, range(1, self.num_classes + 1)), tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        # convert each true label to a vector of ordinal level indicators
        y_true_levels = tf.map_fn(self.label_to_one_hot, y_true)

        # negative sample loss weights
        neg_sample_weights = 0.5 * tf.ones(self.num_classes,
                                           dtype=tf.float32) if self.neg_sample_weights is None else tf.cast(
            self.neg_sample_weights, dtype=tf.float32)
        # class weights
        class_importances = tf.ones(self.num_classes, dtype=tf.float32) if self.class_importances is None else tf.cast(
            self.class_importances, dtype=tf.float32)

        loss = self.get_loss(y_pred, y_true_levels, neg_sample_weights, class_importances)

        return loss

    def get_loss(self, y_pred, y_true_levels, neg_sample_weights, class_importances):
        losses_vec = ((1 - neg_sample_weights) * tf.math.log(
            y_pred + self.epsilon) * y_true_levels + neg_sample_weights * tf.math.log(1 - y_pred + self.epsilon) * (
                                  1 - y_true_levels)) * class_importances
        losses = tf.reduce_mean(-tf.reduce_sum(losses_vec, axis=1))
        return losses

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
            'neg_sample_weights': self.neg_sample_weights,
            'class_importances': self.class_importances,
            'epsilon': self.epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}

