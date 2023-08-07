# coding:utf-8

# 获取attention层的输出，并保存到离线文件方便后续保存到hive表
# 参考https://blog.csdn.net/u011590738/article/details/126247488

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import time
import argparse
from datetime import datetime, timedelta
from EGES_model_4 import *
from itertools import chain
from feature_conf import *

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"你好显卡 : {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
print("tf_version: ", tf.__version__)


def get_file_names(path, suffix_names=None):
    """获取目录下的文件名字"""
    file_names = []
    for filename in os.listdir(path):
        if filename.startswith('part'):
            if suffix_names:
                for suffix_name in suffix_names:
                    if filename.endswith(suffix_name):
                        path1 = os.path.join(path, filename)
                        file_names.append(path1)
            else:
                path1 = os.path.join(path, filename)
                file_names.append(path1)
    return file_names


def process_array_col(array_col):
    res = []
    cols = [int(i) for i in str(array_col).split(',')]
    res.extend(cols)
    length = len(res)
    for i in range(length, 3):
        res.extend([-3])
    return np.array(res)


def load_final_model(final_model_path, lr, embedding_dim,
                     num_nodes, n_sampled, feature_names):
    """修改默认值，输入和输出，保存到线上指定的目录"""
    # 销毁当前的TF图并创建一个新图, 有助于避免旧模型/图层混乱
    tf.keras.backend.clear_session()
    tfra.dynamic_embedding.enable_inference_mode()
    new_model = build_model(
        lr=lr,
        embedding_dim=embedding_dim,
        is_training=False,
        num_nodes=num_nodes,
        n_sampled=n_sampled,
        feature_names=feature_names
    )
    new_model.load_weights(final_model_path)
    new_model.summary()

    # 删除优化器参数
    emb_names = ["UnifiedDynamicEmbedding", "DynamicWeightEmbedding"]
    for emb_name in emb_names:
        new_model.get_layer(emb_name).optimizer_vars = None
        new_model.get_layer(emb_name)._delete_tracking("optimizer_vars")

    outputs = tf.math.l2_normalize(new_model.get_layer('tf.math.truediv').output,
                                   axis=1,
                                   epsilon=1e-10,
                                   name='geek_l2_norm')
    new_inputs = []
    for input_fea in new_model.inputs:
        if input_fea.name == 'labels':
            continue
        new_inputs.append(input_fea)
    print("new_inputs: ", new_inputs)

    final_model = tf.keras.Model(inputs=new_inputs, outputs=outputs)
    final_model.summary()
    return final_model


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_sampled", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--embedding_dim", type=int, default=128)
    # 需要指定路径
    parser.add_argument("--root_path", type=str,
                        default='/alluxio/arc8-brcd-expr-dataset/liuchen04/get/eges/answer/2023-07-26/')
    parser.add_argument("--emb_save_path", type=str,
                        default='/group/recall/liuchen04/output/emb_test.csv')
    parser.add_argument("--final_model_path", type=str, default="/group/recall/liuchen04/model/model_v1")

    args = parser.parse_args()
    print(f'dim:{args.embedding_dim}, batch_size:{args.batch_size}, '
          f'n_sampled:{args.n_sampled}, epochs:{args.epochs}')

    print("load files......")
    # side_info_file_name = "/group/live/liuchen04/get/eges/data/side_info.csv"
    side_info_file_name = get_file_names(args.root_path + 'side_info_30')[0]
    print('side_info文件:', side_info_file_name)
    print(f'连续特征: {len(static_numeric_dict_c.keys())}, 离散特征:{len(static_cate_dict_c)}, 序列特征:{len(static_array_fea_c)}')
    file_data_df = pd.read_csv(side_info_file_name, delimiter=',', usecols=feature_names_c)
    feature_names_order = []
    lisan_feature = [i for i in static_cate_dict_c]
    array_feature = [i for i in static_array_fea_c]
    lianxu_feature = [i for i in static_numeric_dict_c.keys()]
    for feature in chain(lisan_feature, array_feature, lianxu_feature):
        feature_names_order.append(feature)
    # 防止有空值，处理序列特征
    df = file_data_df[feature_names_order].fillna(-3)
    for fea in static_array_fea_c:
        df[fea] = df[fea].apply(lambda x: process_array_col(x))
    df = df.set_index('geek_id', drop=False)  # 设置索引列
    num_nodes = df.shape[0]
    num_features = df.shape[1]
    feature_names = df.columns

    print("Load the final model and predict the embedding...")
    final_model = load_final_model(args.final_model_path, args.lr,
                                   args.embedding_dim, num_nodes,
                                   args.n_sampled, feature_names)
    print("df.count: ", len(df))

    new_df = df.to_dict('list')
    inputs_dict = {}
    for key, value in new_df.items():
        if key not in static_array_fea_c:
            inputs_dict[key] = np.expand_dims(np.array(value), axis=1)
        else:
            inputs_dict[key] = np.array(value)
    pred = final_model(inputs_dict, training=False)
    pred = pred.numpy()
    pred_res = pred[0]
    #     print(pred_res)

    with open(args.emb_save_path, 'w') as f:
        for i in range(len(df)):
            #             print("i: ", i)
            geek_id = new_df['geek_id'][i]
            pred_res = pred[i]
            res = ','.join([str(x) for x in list(pred_res)])
            #             print("res: ", res)
            #             print(str(geek_id) + '\t' + res +'\n')
            f.write(str(geek_id) + '\t' + res + '\n')

            #             if i > 10:
            #                 break

            if i % 10000 == 0:
                print(f"{i} samples saved in file")

    end_time = time.time()
    cost_time = (end_time - start_time) / 60
    print(f"Total cost {cost_time} min in predict")
    print("Finished")
