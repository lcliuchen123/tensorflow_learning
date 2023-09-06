# coding:utf-8

# 使用rnn进行文本分类

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# 用于显示一个耗时操作显示出来的百分比，进度条可以动态的显示进度
tfds.disable_progress_bar()

import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


# 设置输入管道
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
print(train_dataset.element_spec)

for example, label in test_dataset.take(1):
    print("text: ", example.numpy())
    print("label: ", label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
for example, label in train_dataset.take(1):
    print("texts: ", example.numpy()[:3])
    print()
    print("labels: ", label.numpy()[:3])

# 创建文本编码器
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
vocab = np.array(encoder.get_vocabulary())
print(vocab[:20])

encode_example = encoder(example)[:3].numpy()
print(encode_example)

# TextVectorization使用默认设置是不可逆的，主要原因有:
# 1. standardize参数的默认值为"lower_and_strip_punctuation"
# 2. 有限的词汇量和缺乏基于字符的后备导致一些未知的标记

for n in range(3):
    print("original: ", example[n].numpy())
    print("roud-trip: ", " ".join(vocab[encode_example[n]]))
    print()

# 构建双向lstm模型
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
print([layer.supports_masking for layer in model.layers])

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions[0])

padding = "the " * 2000
predictions = model.predict(np.array([sample_text, padding]))
print(predictions[0])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 训练模型
# history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)
# test_loss, test_acc = model.evaluate(test_dataset)
# print("Test Loss: ", test_loss)
# print("Test Acc: ", test_acc)


# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None, 1)
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')
# plt.ylim(0, None)
# plt.show()

# 堆叠2个或者多个lstm
# return_sequences=True返回每一步的h_t
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)
test_loss, test_acc = model.evaluate(test_dataset)
print("Test loss: ", test_loss)
print("Test acc: ", test_acc)

# predict on a sample text without padding.
sample_text = ('The movie was not good. The animation and the graphics '
               'were terrible. I would not recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.show()



