# coding:utf-8

import tensorflow as tf
import numpy as np
from keras import backend as K
from prepare_data import *


print(tf.version.VERSION)


def test_print_layer_output():
    """
       测试tf.keras.backend.print_tensor
       定义打印操作后必须使用该张量，否则不会打印，详情见https://www.qiniu.com/qfans/qnso-43448029#comments
    """
    x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = tf.constant([1, 0, 0, 1])
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat(10).batch(4)
    inputs = tf.keras.layers.Input(shape=(3,))
    inputs = K.print_tensor(inputs, message='\n======inputs_inputs_inputs======:', summarize=-1)
    # K.get_value(inputs)
    print("inputs: ", inputs)
    dense1 = tf.keras.layers.Dense(units=32, activation='relu')(inputs)
    dense1 = K.print_tensor(dense1, message='\n======dense1_dense1_dense1======:', summarize=-1)
    # K.get_value(dense1)
    # print("y: ", y)
    dense2 = tf.keras.layers.Dense(units=32, activation='relu')(dense1)
    dense2 = K.print_tensor(dense2, message='\n======dense2_dense2_dense2======:', summarize=-1)
    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC']
                  )
    model.fit(train_dataset, epochs=3, validation_data=train_dataset)


def DNN(inputs, hidden_units_list):
    """定义DNN层"""
    layer_num = len(hidden_units_list)

    if layer_num <= 0:
        return
    output = tf.keras.layers.Dense(units=hidden_units_list[0],
                                   activation='relu')(inputs)
    for i in range(1, layer_num):
        output = tf.keras.layers.Dense(units=hidden_units_list[i],
                                       activation='relu')(output)
    return output


# todo
def process(inputs):
    """预处理层"""

    return inputs


def build_model(inputs, num_tasks, num_experts, output_info):
    """自定义mmoe"""
    final_outputs = {}
    embedding = process(inputs)

    # 自定义专家
    expert_output_list = []
    for i in range(num_experts):
        hidden_units_list = [256, 128, 64]
        # [batch_size, hidden_units_list[-1]
        single_expert_output = DNN(embedding, hidden_units_list)
        expert_output_list.append(single_expert_output)

    # 合并专家到一块，方便后续使用 [batch_size, num_experts, dim]
    all_expert = tf.stack(expert_output_list, axis=1)

    # 定义每个任务塔
    for i in range(num_tasks):
        # 定义每个任务的gate
        gate_hidden_units_list = [256, 128, 64]
        dnn_output = DNN(embedding, gate_hidden_units_list)
        # [batch_size, num_experts]
        single_task_gate = tf.keras.layers.Dense(units=num_experts,
                                                 activation='softmax')(dnn_output)
        # 扩充维度，[batch_size, num_experts] 变为[batch_size, num_experts, 1]方便后续的点乘
        single_task_gate = tf.expand_dims(single_task_gate, -1)

        # 每个任务的输入
        # 点乘 [batch_size, num_experts, dim]
        single_tower_inputs = tf.multiply(all_expert, single_task_gate)
        single_task_inputs = tf.reduce_sum(single_tower_inputs, axis=1)
        task_hidden_units_list = [256, 128, 64]
        task_output = DNN(single_task_inputs, task_hidden_units_list)
        final_task_output = tf.keras.layers.Dense(units=1,
                                                  activation='sigmoid',
                                                  name=output_info[i])(task_output)
        final_outputs[output_info[i]] = final_task_output

    model = tf.keras.Model(inputs=inputs, outputs=final_outputs)
    return model


def train():
    # step1 准备数据 生成tfrecords文件 / 读取csv文件返回dataset / list或者字典
    get_data()

    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(exponential_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # step2 读取数据
    # step3 数据预处理
    # step4 搭建模型

    # step5 训练模型
    # step6 保存模型
    # step7 模型预测
    # step8 模型保存
    # step9 模型微调
    # step10 修改模型的输入和输出
    pass


if __name__ == "__main__":
    train()
