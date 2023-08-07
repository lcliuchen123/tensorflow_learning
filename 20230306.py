# coding:utf-8

# 保存和恢复模型

import os
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

tf.keras.models.load_model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0


def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


model = create_model()
model.summary()

# 保存为checkpoint
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])


# 只要两个模型共享相同的架构，就可以共享权重
# 重新构建一个新模型
model = create_model()
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# 从checkpoint文件加载权重，并重新评估
model.load_weights(checkpoint_path)

# 重新评估模型
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# checkpoint回调选项,默认只保留最近的5个检查点
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size
)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))

model.fit(train_images, train_labels, epochs=50, batch_size=batch_size,
          callbacks=[cp_callback], validation_data=(test_images, test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

model = create_model()
model.load_weights(latest)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# 手动保存权重
model.save_weights('./checkpoints/my_checkpoint')

model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

# 评估模型
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# 保存整个权重 pb文件
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('old_saved_model/my_model')

new_model = tf.keras.models.load_model('old_saved_model/my_model')

# 检查模型结构
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
print(new_model.predict(test_images).shape)


# hdf5格式, 可能会报错‘str‘ object has no attribute ‘decode‘, 重新下载低版本
# pip install 'h5py<3.0.0' -i https://pypi.tuna.tsinghua.edu.cn/simple

model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')

new_model = tf.keras.models.load_model('my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))




