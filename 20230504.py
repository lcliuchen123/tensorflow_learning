# coding:utf-8

import tensorflow as tf

# 自定义张量

# print(tf.math.add(1, 2))
# print(tf.math.add([1, 2], [3, 4]))
# print(tf.math.square(5))
# print(tf.math.reduce_sum([1, 2, 3]))
#
# print(tf.math.square(2) + tf.math.square(3))
#
# # 矩阵乘法
# x = tf.linalg.matmul([[1]], [[2, 3]])
# print(x)
# print(x.shape)
# print(x.dtype)
#
# # TensorFlow 操作自动将 NumPy ndarrays 转换为 Tensors。
# # NumPy 操作自动将张量转换为 NumPy ndarray
#
# import numpy as np
#
# ndarray = np.ones([3, 3])
# tensor = tf.math.multiply(ndarray, 42)
# print(tensor)
# print(np.add(tensor, 1))
# print(tensor.numpy())
#
# # GPU加速
# x = tf.random.uniform([3, 3])
# print(tf.config.list_physical_devices("GPU"))
#
#
# # 使用上下文管理器将 TensorFlow 操作显式放置在特定设备上tf.device
# import time
#
#
# def time_matmual(x):
#     start = time.time()
#     for loop in range(10):
#         tf.linalg.matmul(x, x)
#     result = time.time() - start
#
#     print("10 loops: {:0.2f}ms".format(1000*result))
#
#
# print("On CPU")
# with tf.device("CPU:0"):
#     x = tf.random.uniform([1000, 1000])
#     assert x.device.endswith('CPU:0')
#     time_matmual(x)
#
# # tf.data,Dataset.map/batch/shuffle
# ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
# print(ds_tensors)
#
# for x in ds_tensors:
#     print(x)


# 自定义图层

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

print(layer(tf.zeros([10, 5])))
print(layer.variables)
print(layer.kernel)
print(layer.bias)

# 自定义层
# 自行实现层的最佳方式是扩展 tf.keras.Layer 类并实现：
# __init__：您可以在其中执行所有与输入无关的初始化
# build：您可以在其中获得输入张量的形状，并可以进行其余初始化
# call：您可以在其中进行前向计算


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]),
                                                       self.num_outputs])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)


layer = MyDenseLayer(10)
_ = layer(tf.zeros([10, 5]))
print([var.name for var in layer.trainable_variables])


tf.data.experimental.make_csv_dataset()
