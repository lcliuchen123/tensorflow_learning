# coding:utf-8

# 实现mmoe模型

import tensorflow as tf
import numpy as np
from data_pipeline import make_dataset
from mmoe import *


def process_feature():
    """预处理层"""
    # 类别 / 非类别
    # 序列
    tf.keras.layers.Input()
    process_inputs = []
    return process_inputs


def build_model(inputs, output_info):
    # todo embedding层 对特征进行预处理
    real_inputs = process_feature(inputs)
    process_layer = tf.keras.layers.DenseFeatures(real_inputs)

    # mmoe层
    mmoe_layers = MMoE(units=4, num_experts=4, num_tasks=4)(process_layer)

    output_layers = {}
    for index, task_layer in enumerate(mmoe_layers):
        label_name = ''
        tower_layer = tf.keras.layers.Dense(units=8,
                                            activation='relu',
                                            kernel_initializer=tf.keras.initializers.variance_scaling())(task_layer)

        output_layer = tf.keras.layers.Dense(units=1,
                                             name=output_info[index],
                                             activation='softmax',
                                             kernel_initializer=tf.keras.initializers.variance_scaling())(tower_layer)
        output_layers[label_name] = output_layer

    model = tf.keras.Model(inputs=inputs, outputs=output_layers)
    return model


def train(asena_path, log_dir, batch_size, model_path):
    # 构造输入
    train_tf_records = asena_path + "/train"
    test_tf_records = asena_path + "/test"
    train_dataset = make_dataset(train_tf_records, no_threads=8, data_type="train")
    test_dataset = make_dataset(test_tf_records, no_threads=8, data_type="test")

    # 构建模型
    inputs = {}
    output_info = ['is_click', 'is_like', 'is_get', 'is_comment']
    model = build_model(inputs, output_info)

    # 定义优化器, 学习率衰减
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(exponential_decay)

    # 在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(optimizer=optimizer,
                  loss={'is_click': 'binary_crossentropy', 'is_like': 'binary_crossentropy',
                        'is_get': 'binary_crossentropy', 'is_comment': 'binary_crossentropy'},
                  loss_weights={'is_click': 2, 'is_like': 1, 'is_get': 1, 'is_comment': 1},
                  metrics={'is_click': tf.keras.metrics.AUC(), 'is_like': tf.keras.metrics.AUC(),
                           'is_get': tf.keras.metrics.AUC(), 'is_comment': tf.keras.metrics.AUC()})

    # 训练模型
    # 添加tensorboard_callback 需要设置profile_batch=0，tensorboard页面才会一直保持更新
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq=batch_size * 200,
        embeddings_freq=1,
        profile_batch=0)

    model.fit(train_dataset,
              validation_data=test_dataset,
              epochs=10,
              callbacks=[tensorboard_callback]
    )

    # 保存模型
    model.save(model_path)


