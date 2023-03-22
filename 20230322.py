# coding:utf-8

# 预处理csv数据

import functools
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

# 让 numpy 数据更易读。
np.set_printoptions(precision=3, suppress=True)

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

# dataset = tf.data.experimental.make_csv_dataset(
#      ...,
#      column_names=CSV_COLUMNS,
#      ...)
# 也可以选择指定的列，select_columns参数


LABEL_COLUMN = 'survived'
LABELS = [0, 1]


def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True
    )
    return dataset


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
examples, labels = next(iter(raw_train_data))  # 第一个批次
print("EXAMPLES: \n", examples, '\n')
print("LABELS: \n", labels)

# 类别数据预处理
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))


# 连续数据预处理
def process_continuous_data(mean, data):
    # 标准化数据
    data = tf.cast(data, tf.float32) * 1 / (2*mean)
    return tf.reshape(data, [-1, 1])


MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

numerical_columns = []

# tf.feature_columns.numeric_column API 会使用 normalizer_fn 参数。
# 在传参的时候使用 functools.partial，functools.partial 由使用每个列的均值进行标准化的函数构成。
for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(
        feature,
        normalizer_fn=functools.partial(process_continuous_data,
        MEANS[feature]))
    numerical_columns.append(num_col)

# 将两列特征集合相加，并传给预处理层
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)

# 构建模型
model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_data = raw_train_data.shuffle(500)
test_data = raw_test_data
model.fit(train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)
print("\n\nTest Loss {}, Test Accuracy {}".format(test_loss, test_accuracy))

predictions = model.predict(test_data)

for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))

# tf.data加载numpy数据, tf加载pandas读取的数据, tf.data.Dataset.from_tensor_slices
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

print(type(train_examples))
print(type(train_labels))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# 可以利用pandas的列代替特征列feature_column
df = pd.read_csv('file.csv')
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)
x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)
model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# 与tf.data一起使用时，最简单方法是转换为dict
target = df.pop('target')
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)
for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)

