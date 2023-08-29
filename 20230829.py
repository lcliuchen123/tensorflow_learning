# coding:utf-8

# 文本词嵌入
import io
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

# 删除无用的文件夹
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# tf.keras.utils.text_dataset_from_directory
batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed
)
val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed
)

for text_batch, label_batch in train_ds.take(1):
    for i in range(5):
        print(label_batch[i].numpy(), text_batch.numpy()[i])


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 1000个单词，5维
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# 嵌入整数
result = embedding_layer(tf.constant([1,2,3]))
print(result.numpy())


# 文本预处理
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')  # 替换特殊字符


vocab_size = 10000
sequence_length = 100
vectorize_layer = TextVectorization(standardize=custom_standardization,
                                    max_tokens=vocab_size,
                                    output_mode='int',
                                    output_sequence_length=sequence_length)
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# 模型
# 该GlobalAveragePooling1D层通过对序列维度进行平均，为每个示例返回固定长度的输出向量。这允许模型以最简单的方式处理可变长度的输入。


embedding_dim = 16
model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name='embedding'),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])
model.summary()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])

# 获取权重矩阵
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

# 保存到文件
out_v = io.open('vector.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0:
        continue
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')
out_m.close()
out_v.close()
