# coding:utf-8

# 基本文本分类, 电影评论

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(test_data)))
print(train_data[0])
print(train_labels[0])
print(len(train_data[0]), len(test_data[0]))

# 把数字索引转换为文本
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = dict((value, key) for (key, value) in word_index.items())


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# 文本评论的长度不一致，输入模型时需要进行调整，保持一致
# 方法一：one-hot编码；方法二：max_length * num_size (本代码采用)

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=256)
print(len(train_data[0]))
print(len(test_data[0]))
print(train_data[0])


# 构建模型
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 优化器和损失函数
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 验证集合
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# 训练模型
history = model.fit(partial_x_train, partial_y_train,
                    epochs=40, batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 模型评估
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

# 创建准确率和损失函数值随时间变化的图像
history_dict = history.history
print(history_dict.keys())

acc = history_dict["accuracy"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

epochs = range(1, len(acc) + 1)

# bo代表蓝点
plt.plot(epochs, loss, 'bo', label="Training Loss")

# b 代表蓝色实线
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 清除数字
plt.clf()

plt.plot(epochs, acc, 'bo', label="Training Accuracy")
plt.plot(epochs, val_accuracy, 'b', label="Validation Accuracy")
plt.title("Training and validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




