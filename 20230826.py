# coding:utf-8

# 图像数据增加: 比如将原来的图像随机旋转产生新样本

# keras预处理层
# tf.image

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# 下载数据集
(train_ds, test_ds, valid_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)
num_classes = metadata.features['label'].num_classes
print(num_classes)

# 随机展示一张图片
get_label_name = metadata.features['label'].int2str
image, label = next(iter(train_ds))
plt.imshow(image)
plt.title(get_label_name(label))
plt.show()

# 利用keras预处理层
IMG_SIZE = 180
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

result = resize_and_rescale(image)
plt.imshow(result)
plt.show()

# 验证像素是否在0～1的范围内
print("min and max values: ", result.numpy().min(), result.numpy().max())

# 数据增强RandomFlip 随机翻转  RandomRotation 随机旋转
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2)
])

image = tf.cast(tf.expand_dims(image, 0), tf.float32)
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(augmented_image[0])
    plt.axis('off')
plt.show()

# 预处理层的两种处理方式
# 方式一: 使预处理层作为模型的一部分，利用gpu进行加速
# model = tf.keras.Sequentail([
#     data_augmentation,
#     resize_and_rescale,
#     layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D()
# ])

# 方式二: 将预处理层应用于数据集
# train_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False, augmentation=False):
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    if augmentation:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(batch_size)


train_ds = prepare(train_ds, shuffle=True, augmentation=True)
test_ds = prepare(test_ds)
valid_ds = prepare(valid_ds)

model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
epochs = 5
history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs)

loss, acc = model.evaluate(test_ds)
print("accuracy: ", acc)


# 自定义数据增强
def random_invert_img(x, p=0.5):
    """以某个概率随机修改颜色"""
    if tf.random.uniform([]) < p:
        x = (255 - x)
    return x


def random_invert(factor=0.5):
    return layers.Lambda(lambda x: random_invert_img(x, factor))
random_invert = random_invert()

for i in range(9):
    augmented_image = random_invert(image)
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(augmented_image[0].numpy().astype('uint8'))
    plt.axis('off')
plt.show()


# 子类化自定义层
class RandomInvert(layers.Layer):
    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return random_invert_img(x)

plt.imshow(RandomInvert()(image)[0])
plt.show()
