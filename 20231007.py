# coding:utf-8

# 不平衡数据的处理: 1.设置正确的初始偏差; 2.类权重; 3.过采样:对数量较少的类别数据进行采样


import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 预处理层
from tensorflow.keras.layers.experimental import preprocessing

# print(tf.data.Dataset.from_tensor_slices.__doc__)
# print(tf.keras.layers.concatenate.__doc__)


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# 数据处理与预览

file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
print(raw_df.head())
print(raw_df.describe())

# 检查类别的不平衡
neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print("Examples:\n Total: {}\n Positive: {} ({:.2f}% of total)\n".format(
    total, pos, 100 * pos / total))

# 清理，删除，归一化数据
cleaned_df = raw_df.copy()
cleaned_df.pop('Time')
eps = 0.001
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# 归一化
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.fit_transform(val_features)
test_features = scaler.fit_transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print("Training labels shape: ", train_labels.shape)
print("Testing labels shape: ", test_labels.shape)
print("Validation labels shape: ", val_labels.shape)

print("Training features shape: ", train_features.shape)
print("Testing features shape: ", test_features.shape)
print("Validation features shape: ", val_features.shape)

# 查看数据分布
pos_df = pd.DataFrame(train_features[bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)
sns.jointplot(x=pos_df['V5'], y=pos_df['V6'], kind='hex', xlim=(-5, 5), ylim=(-5, 5))
plt.suptitle('Positive distribution')
sns.jointplot(x=neg_df['V5'], y=neg_df['V6'], xlim=(-5, 5), ylim=(-5, 5))
_ = plt.suptitle('Negative distribution')


# 定义模型和指标
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR')
]


def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    return model


EPOCHS = 100
BATCH_SIZE = 2048
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)


model = make_model()
model.summary()
print(model.predict(train_features[:10]))


# 设置正确的输出偏差, 在输出层添加偏差，反应这种不平衡
initial_bias = np.log([pos / neg])
print("initial_bias: ", initial_bias)

model = make_model(output_bias=initial_bias)
print(model.predict(train_features[:10]))

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("loss: ", results[0])

# 为初始权重设置检查点
initial_weights = os.path.join(tempfile.mktemp(), 'initial_weights')
model.save_weights(initial_weights)

# 确认权重修正有帮助
model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0
)


model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0
)


def plot_loss(history, label, n):
    plt.semilogy(history.epoch,
                 history.history['loss'],
                 color=colors[n],
                 label='Train ' + label)
    plt.semilogx(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val '+label, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


plot_loss(zero_bias_history, 'ZERO_BIAS', 0)
plot_loss(careful_bias_history, 'careful_BIAS', 1)
plt.show()

# 训练模型
model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels)
)


# 查看训练历史记录
def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        # 首字母大写
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2,2, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()


plot_metrics(baseline_history)


# 混淆矩阵
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > 0.5)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('confusion_matrix @{:.2f}'.format(p))
    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
baseline_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ':', value)
print()
plot_cm(test_labels, test_predictions_baseline)


# 绘制roc曲线
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel("False positive [%]")
    plt.ylabel("True positive [%]")
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


# 绘制auprc
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


plot_prc('Train Baseline', train_labels, train_predictions_baseline, color=colors[0])
plot_roc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
plt.show()


# 类权重
weight_for_0 = (1 / neg) * (total) / 2.0
weight_for_1 = (1 / pos) * (total) / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print("class_weight: ", class_weight)

weighted_model = make_model()
weighted_model.load_weights(initial_weights)
weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels),
    class_weight=class_weight
)

plot_metrics(weighted_history)
