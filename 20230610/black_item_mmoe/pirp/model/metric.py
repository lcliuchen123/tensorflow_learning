# coding:utf-8

import tensorflow as tf


class ClassAUC(tf.keras.metrics.Metric):
    def __init__(self, target, target_config, **kwargs):
        super(ClassAUC, self).__init__(**kwargs)
        self.target = target
        self.bins = range(1, target_config['bin_cnt'] + 1)
        self.aucs = {}
        for bin in self.bins:
            self.aucs[str(bin)] = tf.keras.metrics.AUC(name='auc_{0}'.format(bin))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_levels = tf.map_fn(lambda x: tf.cast(tf.greater_equal(x, self.bins), tf.float32), y_true)
        for idx, bin in enumerate(self.bins):
            self.aucs[str(bin)].update_state(y_true_levels[:, idx], y_pred[:, idx])

    def result(self):
        # 2.4版本不支持返回dict，2.8+支持
        aucs = {}
        for bin in self.bins:
            aucs['{0} level >= {1}'.format(self.target, bin)] = self.aucs[str(bin)].result()
        return aucs

