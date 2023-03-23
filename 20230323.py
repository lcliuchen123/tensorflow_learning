# coding:utf-8

# tfrecord和tf.Example

import tensorflow as tf
import numpy as np
import IPython.display as display

# 数据预处理过程
# 写入tfrecords文件
# filename ——> numpy数组 ——> 根据数据类型转化为tf.train.BytesList / FloatList / Int64List
# ——> tf.train.Feature ——> tf.train.Example ——> tf.train.Example.SerializeToString()
# ——> tf.io.TFRecordWriter(filename)

# 读取tfrecords文件
# tf.data.TFRecordDataset(filename) ——> 获取dataset
# tf.io.parse_single_example(example, feature_dict) ——> 获取parse_example函数
# dataset.map(_parse_function)

# tf.Example 是 {"string": tf.train.Feature} 映射


# 处理标量特征，非标量特征采用tf.io.serialize_tensor, tf.io.parse_tensor
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))
print(_float_feature(np.exp(1)))
print(_int64_feature(True))
print(_int64_feature(1))

# 转化为二进制字符串
print(_float_feature(np.exp(1)).SerializeToString())

# 随机生成样本
n_observations = int(1e4)

feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)


def serialize_example(feature0, feature1, feature2, feature3):
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


example_observation = []
serialize_examples = serialize_example(False, 4, b'goat', 0.9876)
# 二进制
print(serialize_examples)
# 解码
example_proto = tf.train.Example.FromString(serialize_examples)
print(example_proto)

# tf.data.Dataset.from_tensor_slices
feature_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
for f0, f1, f2, f3 in feature_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)


def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),
        tf.string
    )
    return tf.reshape(tf_string, ())


serialize_features_dataset = feature_dataset.map(tf_serialize_example)
print(serialize_features_dataset)


def generator():
    for features in feature_dataset:
        yield serialize_example(*features)


serialize_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

# 写入tfrecords文件
filename = 'test.record'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialize_features_dataset)

# 读取tfrecords文件
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

for raw_data in raw_dataset.take(10):
    print(repr(raw_data))

feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
}


def _parase_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


parsed_dataset = raw_dataset.map(_parase_function)
print(parsed_dataset)

# 获取每个标量
for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))


# tf.io写tfrecord文件
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

# tf.io读tfrecord文件
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)


# 读取和写入图像数据
cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

print(display.display(display.Image(filename=cat_in_snow)))
display.display(display.HTML('Image cc-by: &lt;a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg"&gt;Von.grzanka&lt;/a&gt;'))

# 写入tfrecords文件
image_labels = {
    cat_in_snow: 0,
    williamsburg_bridge: 1
}

image_string = open(cat_in_snow, 'rb').read()
label = image_labels[cat_in_snow]


def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)
print("....")

# 写入tfrecord文件
record_file = 'image.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())

# 读取tfrecords文件
raw_image_dataset = tf.data.TFRecordDataset(record_file)

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}


def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset)

for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    display.display(display.Image(data=image_raw))





