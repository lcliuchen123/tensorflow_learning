# coding: utf-8

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

# 回归问题，画图有点问题
print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

row_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = row_dataset.copy()
print(dataset.tail())

# 数据清洗
print(dataset.isna().sum())

# 删除空值列
dataset = dataset.dropna()

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 2.0
dataset['Japan'] = (origin == 3) * 3.0
print(dataset.tail())


# 拆分训练数据集和测试数据集
train_dataset = dataset.sample(frac=0.8, random_state=0)
train_dataset = dataset.drop(train_dataset.index)
test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# 统计
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# 从标签中分离特征
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


# 标准化
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 构造模型
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


model = build_model()
model.summary()


# 测试模型
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(" ")
        print('.', end='')


EPOCHS = 1000
history = model.fit(normed_train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


# def plot_history(history):
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch
#     plt.figure()
#     plt.xlabel("Epoch")
#     plt.ylabel("MEAM ABS ERROR FOR [MPG] ")
#     plt.plot(hist['epoch'], hist['mae'], label="Train Error")
#     plt.plot(hist['epoch'], hist['val_mae'], label="Val Error")
#     plt.ylim([0, 5])
#     plt.legend()
#
#     plt.figure()
#     plt.xlabel("Epoch")
#     plt.ylabel("Mean Square Error [$MPG^2$]")
#     plt.plot(hist['epoch'], hist['mse'], label='Train Error')
#     plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
#     plt.ylim([0, 20])
#     plt.legend()
#     plt.show()


# plot_history(history)


# 早停
model = build_model()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, verbose=1,
                    validation_split=0.2, callbacks=[early_stop, PrintDot()])


loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set mean abs error: {:5.2f} MPG".format(mae))

# 预测
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel("TRAIN VALUES [MPG]")
plt.ylabel("PREDICTIONS [MPG]")
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
# plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel('Count')
