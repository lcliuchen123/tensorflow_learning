# coding: utf-8

# deepfm & ppnet
# 添加文本特征，训练模型
import time
import argparse
from model import *
from hive_to_asena_v2 import *


def get_fea_type(fea_col_dict):
    """定义需要解析的特征格式"""
    schema = {}
    for fea_name, fea_type in fea_col_dict.items():
        if fea_name == 'deal_type':
            continue

        if fea_name in seq_feature_length_dict:
            length = seq_feature_length_dict[fea_name]
            if fea_type in ('int-vector', 'long-vector'):
                schema[fea_name] = tf.io.FixedLenFeature(dtype=tf.int64, shape=(length,))
            else:
                schema[fea_name] = tf.io.FixedLenFeature(dtype=tf.float32, shape=(length,))
        else:
            if fea_type in ('int', 'long'):
                schema[fea_name] = tf.io.FixedLenFeature(dtype=tf.int64, shape=(1,))
            else:
                schema[fea_name] = tf.io.FixedLenFeature(dtype=tf.float32, shape=(1,))
    return schema


def _parse_deepfm_feature(example):
    """解析tf-records的特征"""
    # 特征名字和类型表
    file_path = '/group/live/liuchen04/cp/20231211/特征.csv'
    f1_columns_dict, new_columns_dict, text_columns_dict, _ = get_features(file_path)
    all_schema = {}
    # job侧特征
    f1_schema = get_fea_type(f1_columns_dict)
    all_schema.update(f1_schema)
    # position侧特征
    new_schema = get_fea_type(new_columns_dict)
    all_schema.update(new_schema)
    # 文本相关特征
    text_schema = get_fea_type(text_columns_dict)
    all_schema.update(text_schema)
    print("all_schema: ", all_schema)
    feature = tf.io.parse_single_example(example, all_schema)

    schema = {}
    schema = {"is_addf": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1,))}
    labels = tf.io.parse_single_example(example, schema)
    return feature, labels


def _parse_ppnet_feature(example):
    """解析tf-records的特征"""
    # 特征名字和类型表
    file_path = '/group/live/liuchen04/cp/20231211/特征.csv'
    f1_columns_dict, new_columns_dict, text_columns_dict, _ = get_features(file_path)
    all_schema = {}
    # job侧特征
    f1_schema = get_fea_type(f1_columns_dict)
    all_schema.update(f1_schema)
    # position侧特征
    new_schema = get_fea_type(new_columns_dict)
    all_schema.update(new_schema)
    # 文本相关特征
    text_schema = get_fea_type(text_columns_dict)
    all_schema.update(text_schema)
    print("all_schema: ", all_schema)
    feature = tf.io.parse_single_example(example, all_schema)

    schema = {}
    schema = {"is_addf": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1,)),
              "is_success": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1,))}
    labels = tf.io.parse_single_example(example, schema)
    return feature, labels


def get_sample_data(tf_record_path, today, data_type, model_type):
    """读取tf-records文件获取训练集和测试集"""
    # step1 定义解析tfrecords的features和label的函数
    day_list = get_day_list(today, data_type)

    # 测试
    if data_type == 'local_test':
        day_list = ['2024-03-11']
        tf_record_path = "/group/live/liuchen04/cp/20231211/data/"

    # step2 读取文件路径生成文件列表
    print("day_list: ", day_list)
    file_names = []
    for day in day_list:
        files = tf.data.Dataset.list_files(tf_record_path + "/" + day + "/*tfrecord.gz", shuffle=True)
        file_names += files

    # step3 创建dataset
    dataset = tf.data.TFRecordDataset(file_names, compression_type='GZIP', num_parallel_reads=8)

    # step4 解析特征和label
    if model_type == 'deepfm':
        dataset = dataset.map(_parse_deepfm_feature)
    elif model_type == 'ppnet':
        dataset = dataset.map(_parse_ppnet_feature)
    else:
        print("INPUT MSST deepfm or ppnet")
        return None
    return dataset


def save_model(model, save_path):
    """最终保存模型到指定的路径并修改模型的输入输出"""
    pass


def main(today, train_data_path, test_data_path, embedding_dim,
         batch_size, lr, epochs, max_epochs_without_improve,
         model_type, model_path, log_dir):
    print("step1: 获取样本数据")
    dataset = get_sample_data(train_data_path, today, "train", model_type)

    # 测试
    #     dataset = get_sample_data(train_data_path, today, "valid", model_type)
    train_dataset = dataset.shuffle(buffer_size=batch_size).batch(batch_size). \
        prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for index, features in enumerate(train_dataset):
        if index > 1:
            break
        print("features: ", features)

    test_dataset = get_sample_data(test_data_path, today, "test", model_type)

    # 测试
    #     test_dataset = get_sample_data(test_data_path, today, "valid", model_type)

    test_dataset = test_dataset.batch(batch_size)

    print("step2: 构造模型")
    model = sim_cp_model(model_type, embedding_dim)

    print("step3: 训练模型")
    # 配置相关学习参数
    if model_type == 'deepfm':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss={"is_addf": "binary_crossentropy"},
                      metrics={"is_addf": tf.keras.metrics.AUC()})
    elif model_type == 'ppnet':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss={"is_addf": "binary_crossentropy", "is_success": "binary_crossentropy"},
                      loss_weights={"is_addf": 1, "is_success": 1},
                      metrics={"is_addf": tf.keras.metrics.AUC(), "is_success": tf.keras.metrics.AUC()})
    else:
        print("Must input the correct model type: deepfm or ppnet")

    # 自定义callbacks
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=max_epochs_without_improve,
                                                      min_delta=0.1,
                                                      verbose=1)

    # 训练模型
    model.fit(train_dataset,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=test_dataset,
              callbacks=[tensorboard_callbacks, early_stopping])

    # 自定义模型没有定义输入的形状，无法打印模型
    model.summary()

    print("step4: 保存模型, pb文件")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    tf.keras.models.save_model(model, model_path)

    # 修改模型的输入和输出，保存为最终的样子，推送到线上
    # save_model(model, model_path)


if __name__ == '__main__':
    start_time = time.time()
    parse = argparse.ArgumentParser()
    parse.add_argument("--today", type=str, default="2023-12-12")
    parse.add_argument("--train_data_path", type=str, default="")
    parse.add_argument("--test_data_path", type=str, default="")
    parse.add_argument("--embedding_dim", type=int, default=32)
    parse.add_argument("--batch_size", type=int, default=1024)
    parse.add_argument("--lr", type=float, default=0.0001)
    parse.add_argument("--epochs", type=int, default=10)
    parse.add_argument("--max_epochs_without_improve", type=int, default=3)
    parse.add_argument("--model_type", type=str, default="deepfm")
    parse.add_argument("--model_path", type=str, default="")
    parse.add_argument("--log_dir", type=str, default="")
    args = parse.parse_args()
    main(args.today, args.train_data_path, args.test_data_path, args.embedding_dim,
         args.batch_size, args.lr, args.epochs, args.max_epochs_without_improve,
         args.model_type, args.model_path, args.log_dir)
    end_time = time.time()
    cost_time = (end_time - start_time) / 3600
    print(f"Finished in {cost_time} h")
