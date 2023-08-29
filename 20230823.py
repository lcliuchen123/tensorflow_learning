# coding:utf-8

# 利用 Tensorflow Hub进行迁移学习
# TensorFlow Hub是预训练 TensorFlow 模型的存储库。

import numpy as np
import time
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import datetime

# 利用预训练好的模型生成keras模型
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
classifier_model = mobilenet_v2
IMAGE_SIZE = (224, 224)
classifier = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=IMAGE_SIZE+(3,))])

# 单个图像运行
grace_hopper = tf.keras.utils.get_file("image.jpg", 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SIZE)
grace_hopper = np.array(grace_hopper) / 255.0
print(grace_hopper.shape)

# 添加批量维度
result = classifier.predict(grace_hopper[np.newaxis, ...])
print(result.shape)

predict_class = tf.math.argmax(result[0], axis=-1)
print(predict_class)

# 解码预测
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predict_class_name = imagenet_labels[predict_class]
_ = plt.title("Prediction: " + predict_class_name.title())


# 用自己的数据集自定义分类器
import pathlib
data_file = tf.keras.utils.get_file('flower_photos.tgz',
                                    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                    cache_dir='',
                                    extract=True)
data_root = pathlib.Path(data_file).with_suffix('')
print("data_root: ", data_root)

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_root),
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(data_root),
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = np.array(train_ds.class_names)
print("class_names: ", class_names)

# hub约定输入是[0,1]范围的浮点数
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# 对一批图像上运行分类器
result_batch = classifier.predict(train_ds)
print(result_batch.shape)
predict_class_names = imagenet_labels[tf.math.argmax(result_batch, axis=-1)]
print(predict_class_names)

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n+1)
    plt.imshow(image_batch[n])
    plt.title(predict_class_names[n])
    plt.axis('off')
_ = plt.suptitle("Imagenet predictions")


# Hub没有顶层分类层的模型
# 预训练层
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor_model = mobilenet_v2
feature_extractor_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

num_classes = len(class_names)
model = tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dense(num_classes)])
model.summary()

predictions = model(image_batch)
print(predictions.shape)

# 训练模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
NUM_EPOCHS = 10
model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=tensorboard_callback)

# 预测
predict_batch = model.predict(image_batch)
predict_id = tf.math.argmax(predict_batch, axis=-1)
print("predict_id: ", predict_id)
predict_label_batch = class_names[predict_id]
print(predict_label_batch.shape)
print(predict_label_batch)


# 保存模型
t = time.time()
export_path = "/tmp/saved_models/{}".format(int(t))
print(export_path)
model.save(export_path)

# 加载模型
reloaded = tf.keras.models.load_model(export_path)
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
print(reloaded_result_batch.shape)
print(abs(reloaded_result_batch - result_batch).max())
