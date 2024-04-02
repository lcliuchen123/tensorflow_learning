# coding:utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import os


class TrainingHistory(tf.keras.callbacks.Callback):

    def __init__(self, targets, log_path, has_val=True):
        super().__init__()

        # targets
        self.targets = targets
        # if there is validation data
        self.has_val = has_val

        # save loss
        self.loss = {}
        for target in self.targets:
            self.loss[target] = {'train': [], 'val': []}
        # log dir
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        target_cnt = len(self.targets)
        for target in self.targets:
            self.loss[target]['train'].append(logs.get('{0}_loss'.format(target) if target_cnt > 1 else 'loss'))
            if self.has_val:
                self.loss[target]['val'].append(logs.get('val_{0}_loss'.format(target) if target_cnt > 1 else 'val_loss'))

    def on_train_end(self, logs=None):
        self.__plot_history()

    def __plot_history(self):
        # 参数
        plot_settings = {
            'loss': self.loss
        }
        metric_cnt = len(plot_settings)
        target_cnt = len(self.targets)

        plt.figure(figsize=(4 * metric_cnt, 4 * target_cnt))
        seq = range(len(self.loss[list(self.targets)[0]]['train']))
        colors = ['blue', 'orange', 'green', 'orangered', 'sandybrown']
        for t_i, target in enumerate(self.targets):
            for m_i, metric in enumerate(plot_settings):
                ax = plt.subplot(target_cnt, metric_cnt, t_i * metric_cnt + m_i + 1)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('{0} {1}'.format(target.upper(), metric.upper()))
                plt.grid(True)
                for idx, flag in enumerate(['train', 'val'] if self.has_val else ['train']):
                    ys = plot_settings[metric][target][flag]
                    plt.plot(seq, ys, colors[idx], marker='o')
                    self.__plot_text(ax, seq, ys)
                plt.legend(['train', 'val'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_path, 'history_plot.png'))

    @staticmethod
    def __plot_text(ax, xs, ys):
        for x, y in zip(xs, ys):
            ax.text(x, y, '%.4f' % y, ha='center', va='bottom', fontsize=10)
