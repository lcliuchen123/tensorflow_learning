# coding:utf-8

# Keras Tuner 是一个库，可帮助您为 TensorFlow 程序选择最佳的超参数集。
# 为您的机器学习 (ML) 应用选择正确的超参数集，这一过程称为超参数调节或超调。

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

(img_train, label_train), (img_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()

# 归一化到0～1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

# 两种方式定义超模型：1. 使用模型构建工具函数; 2.将 Keras Tuner API 的 HyperModel 类子类化


# 使用模型构建工具函数
def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    # 隐层单元数32～512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# Keras Tuner 提供了四种调节器：RandomSearch、Hyperband、BayesianOptimization 和 Sklearn。
# 在本教程中，您将使用 Hyperband 调节器。
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# 获取最优超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("best_hps: ", best_hps)
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")


# 训练模型
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print("best epoch : %d" % (best_epoch,))

# 重新实例化进行训练
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

# 评估效果
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]: ", eval_result)
