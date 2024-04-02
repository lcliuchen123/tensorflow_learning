# coding:utf-8

import time
import argparse
import pandas as pd
import tensorflow as tf
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


def _parse_feature(example):
    """解析tf-records的特征"""
    # 特征名字和类型表
    file_path = '特征.csv'
    f1_columns_dict, new_columns_dict, _, _ = get_features(file_path)
    all_schema = {}
    # job侧特征
    f1_schema = get_fea_type(f1_columns_dict)
    all_schema.update(f1_schema)
    # position侧特征
    new_schema = get_fea_type(new_columns_dict)
    all_schema.update(new_schema)
    # 文本相关特征
    print("all_schema: ", all_schema)
    feature = tf.io.parse_single_example(example, all_schema)

    schema = {}
    schema = {"is_addf": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1,)),
              "is_success": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1,))}
    labels = tf.io.parse_single_example(example, schema)

    schema = {}
    schema = {"job_id": tf.io.FixedLenFeature(dtype=tf.int64, shape=(1,))}
    job_id = tf.io.parse_single_example(example, schema)
    return feature, labels, job_id


def get_sample_data():
    """读取tf-records文件获取训练集和测试集"""
    # step1:获取本地测试集
    day_list = ['2023-12-13']
    tf_record_path = "test"
    # step2 读取文件路径生成文件列表
    file_names = []
    for day in day_list:
        files = tf.data.Dataset.list_files(tf_record_path + "/" + day + "/*tfrecord.gz", shuffle=True)
        file_names += files
    # step3 创建dataset
    dataset = tf.data.TFRecordDataset(file_names, compression_type='GZIP', num_parallel_reads=8)

    # step4 解析特征和label
    dataset = dataset.map(_parse_feature)
    return dataset


def main(model_type, model_path, batch_size, local_path):
    print("step1: 读取测试集数据")
    test_dataset = get_sample_data()
    test_dataset = test_dataset.batch(batch_size)

    print("step2: 加载训练好的模型, 预测得分")
    model = tf.keras.models.load_model(model_path)

    print("step3: 保存到指定的本地路径")
    num = 0
    with open(local_path, 'w') as f:
        for index, inputs in enumerate(test_dataset):
            features, labels, job_id = inputs
            score = model(features, training=False)
            job_id = list(job_id['id'].numpy()[:, 0])
            is_addf = list(labels['is_addf'].numpy()[:, 0])
            is_success = list(labels['is_success'].numpy()[:, 0])
            addf_pred = list(score['is_addf'].numpy()[:, 0])
            if model_type == 'ppnet':
                succ_pred = list(score['is_success'].numpy()[:, 0])
            else:
                succ_pred = [-1000] * batch_size
            first_row = ['id', 'is_addf', 'is_success', 'addf_pred', 'succ_pred']
            first_row = '\t'.join(first_row)
            if num == 0:
                f.write(first_row + '\n')

            for c1, c2, c3, c4, c5 in zip(job_id, is_addf, is_success, addf_pred, succ_pred):
                row = [c1, c2, c3, c4, c5]
                row = [str(x) for x in row]
                row = '\t'.join(row)
                f.write(row + '\n')
            num += 1
            if num % 1000 == 0:
                print(f"{num} batch have been saved in {local_path}")
    print("num: ", num)


if __name__ == "__main__":
    start_time = time.time()
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_type", type=str, default="deepfm")
    parse.add_argument("--batch_size", type=int, default=2048)
    parse.add_argument("--model_path", type=str, default="/model_test/2023-12-24-15-53-28")
    parse.add_argument("--local_save_path", type=str, default="test.csv")
    args = parse.parse_args()
    main(args.model_type, args.model_path, args.batch_size, args.local_save_path)
    end_time = time.time()
    cost_time = (end_time - start_time) / 3600
    print(f"Finished in {cost_time} h")


