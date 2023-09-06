# coding:utf-8

# 热启动词嵌入
# 为什么要热启动嵌入矩阵？
# 使用一组表示给定词汇的嵌入来训练模型。如果模型需要更新或改进，
# 您可以通过重用先前运行的权重来训练收敛速度明显更快。使用先前运行的嵌入矩阵更加困难。问题在于，对词汇表的任何更改都会使单词到 id 的映射失效。

# tf.keras.utils.warmstart_embedding_matrix通过根据基础词汇表的嵌入矩阵为新词汇表创建嵌入矩阵来解决这个问题。
# 如果两个词汇表中都存在单词，则基本嵌入向量将被复制到新嵌入矩阵中的正确位置。这使您可以在词汇量或顺序发生任何变化后热启动训练。

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
print(tf.__version__)
# 要求tf 版本为2.11, 不进行展示，详细内容见https://www.tensorflow.org/text/tutorials/warmstart_embedding_matrix

inputs = tf.random.normal([2, 3, 8])
lstm = tf.keras.layers.LSTM(1)
output = lstm(inputs)  # h_t
print(output.shape)
print(output.numpy())
# (2, 4)

lstm = tf.keras.layers.LSTM(1, return_sequences=True, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output.shape)
print(whole_seq_output.numpy())
# (2, 10, 4)
print(final_memory_state.shape)
print(final_memory_state.numpy())
# (2, 4)
print(final_carry_state.shape)
print(final_carry_state.numpy())
# (2, 4)

