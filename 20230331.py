# coding: utf-8

# 文本
# 该部分代码由于tensorflow版本问题，未成功运行

import collections
import pathlib
import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_text as tf_text


# 预测stack overflow 问题的标签
data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset_dir = tf.keras.utils.get_file(
    origin=data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir=''
)

dataset_dir = pathlib.Path(dataset_dir).parent
print(list(dataset_dir.iterdir()))

train_dir = dataset_dir/"train"
print(list(train_dir.iterdir()))


sample_file = train_dir/'python/1755.txt'
with open(sample_file) as f:
    print(f.read())

# 加载数据集 tf.keras.utils.text_dataset_from_directory 从目录下加载文件
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(10):
        print("QUESTION: ", text_batch.numpy()[i])
        print("label: ", label_batch.numpy()[i])

for i, label in enumerate(raw_train_ds.class_names):
    print("Label", i, " corresponds to ", label)

# 验证集
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

test_dir = dataset_dir/'test'
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size
)


# tf.keras.layers.TextVectorization 层对数据进行标准化、词例化和向量化。
# 标准化是指预处理文本，通常是移除标点符号或 HTML 元素以简化数据集。
# 词例化是指将字符串拆分为词例（例如，通过按空格分割将一个句子拆分为各个单词）。
# 向量化是指将词例转换为编号，以便将它们输入到神经网络中。

VOCAB_SIZE = 10000
# 词包
binary_vectorize_layer = tf.keras.utils.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary'
)

# 词序
MAX_SEQUENCE_LENGTH = 250
int_vectorize_layer = tf.keras.utils.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

# 调用 TextVectorization.adapt 以使预处理层的状态适合数据集。这会使模型构建字符串到整数的索引。
train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)


def binary_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return binary_vectorize_layer(text), label


def int_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return int_vectorize_layer(text), label


text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[0], label_batch[0]
print("Question: ", first_question)
print("Label: ", first_label)

print("binary vectorized question: ", binary_vectorize_text(first_question, first_label)[0])
print("int vectorized question: ", int_vectorize_text(first_question, first_label)[0])

# 调用 TextVectorization.get_vocabulary 来查找每个整数对应的词例（字符串）
print("1289 ---> ", int_vectorize_layer.get_vocabulary()[1289])
print("313 ---> ", int_vectorize_layer.get_vocabulary()[313])
print("vocab size: ", len(int_vectorize_layer.get_vocabulary()))

binary_train_ds = raw_train_ds.map(binary_vectorize_text)
binary_val_ds = raw_val_ds.map(binary_vectorize_text)
binary_test_ds = raw_val_ds.map(binary_vectorize_text)

int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

# 配置数据集，防止堵塞 cache, prefetch
AUTOTUNE = tf.data.AUTOTUNE


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)
binary_test_ds = configure_dataset(binary_test_ds)

int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)

# 训练模型
binary_model = tf.keras.Sequential([tf.keras.layers.Dense(4)])
binary_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

history = binary_model.fit(binary_train_ds, validation_data=binary_val_ds, epochs=10)


# int 创建模型
def create_model(vocab_size, num_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
        tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(num_labels)
    ])


int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=4)
int_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)
print("linear model on binary vectorized data")
print(binary_model.summary())

print("ConvNet model on int vectorized data")
print(int_model.summary())

binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
int_loss, int_accuracy = int_model.evaluate(int_test_ds)

print("Binary model accuracy: {:2.2f%}".format(binary_accuracy))
print("int model accuracy: {:2.2f%}".format(int_accuracy))

# 用上面训练的权重重新创建一个模型, 将文本预处理包在模型里面（binary_vectorize_layer）
export_model = tf.keras.Sequential([
    binary_vectorize_layer, binary_model,
    tf.keras.layers.Activation('sigmoid')
])
export_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

loss, accuracy = export_model.evaluate(raw_test_ds)
print("accuracy: {:2.2f%}".format(accuracy))


def get_string_label(predict_scores_batch):
    predict_int_labels = tf.math.argmax(predict_scores_batch, axis=1)
    predict_labels = tf.gather(raw_train_ds.class_names, predict_int_labels)
    return predict_labels


inputs = [
    "how do I extract keys from a dict into a list?",  # 'python'
    "debug public static void main(string[] args) {...}",  # 'java'
]
predict_scores = export_model.predict(inputs)
predict_labels = get_string_label(predict_scores)
for input, label in zip(inputs, predict_labels):
    print("Question: ", input)
    print("Predicted label: ", label)

