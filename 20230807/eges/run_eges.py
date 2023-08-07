# coding:utf-8

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import time
import argparse
import random
# from arsenal_magic.graph.application.worflow_map import WorkflowMap
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


def get_n_days_before(x, n_days):
    x_dt = datetime.strptime(x, "%Y-%m-%d")
    ret_dt = x_dt - timedelta(days=n_days)
    return ret_dt.strftime("%Y-%m-%d")


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


def process_array_col(array_col, batch_size):
    res = []
    try:
        for value in array_col:
            # print(value)
            cols = [int(i) for i in str(value).split(',')]
            res.append(cols)
    except:
        # 为空时会报错，补充为-3，保持长度都为batch_size
        length = len(res)
        for i in range(length, batch_size):
            res.append([-3, -3, -3])
        return np.array(res)
    return np.array(res)


# 随机mask获取默认的embedding
def get_dataset_mask(pair_file_names_train, side_info_file_name):
    print(f'连续特征: {len(static_numeric_dict_c.keys())}, 离散特征:{len(static_cate_dict_c)}, 序列特征:{len(static_array_fea_c)}')
    file_data_df = pd.read_csv(side_info_file_name, delimiter=',', usecols=feature_names_c)
    feature_names_order = []
    lisan_feature = [i for i in static_cate_dict_c]
    array_feature = [i for i in static_array_fea_c]
    lianxu_feature = [i for i in static_numeric_dict_c.keys()]
    for feature in chain(lisan_feature, array_feature, lianxu_feature):
        feature_names_order.append(feature)
    df = file_data_df[feature_names_order]
    df = df.set_index('geek_id', drop=False)  # 设置索引列
    num_nodes = df.shape[0]
    num_features = df.shape[1]
    feature_names = df.columns

    # 对于label进行编码
    dict_lianxu = {i: tf.float32 for i in lianxu_feature}
    dict_array = {i: tf.int64 for i in array_feature}
    dict_lisan = {i: tf.int64 for i in lisan_feature}
    # generator_output_types = {**dict_lisan, **dict_array, **dict_lianxu, **{"geek_id": tf.int64, "labels": tf.int64}}
    # generator_output_shapes = {**{i: [None, 1] for i in lianxu_feature + lisan_feature},
    #                            **{i: [None, 3] for i in array_feature},
    #                            **{"geek_id": [None, 1], "labels": [None, 1]}}

    generator_output_types = ({**dict_lisan, **dict_array, **dict_lianxu, **{"geek_id": tf.int64, "labels": tf.int64}},
                              tf.int64)
    generator_output_shapes = ({**{i: [None, 1] for i in lianxu_feature + lisan_feature},
                                **{i: [None, 3] for i in array_feature},
                                **{"geek_id": [None, 1], "labels": [None, 1]}},
                               [None, 1])

    def raw_data_gen_batch_mask():
        now_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(now_datetime, "epoch开始")
        random_n = 100_0000 // args.batch_size  # 每个
        for file_name in pair_file_names_train:
            chunk_train = pd.read_csv(file_name, chunksize=args.batch_size, header=None)
            for chunk in chunk_train:
                if len(chunk.values) == args.batch_size:
                    x = chunk.values[:, 1]
                    y = np.expand_dims(x, axis=1)
                    my_dict = {'labels': y}
                    labels = y
                    batch_side_info = df.loc[chunk.values[:, 0]]
                    for i, col in enumerate(df.columns):
                        x = batch_side_info[col]
                        if col in array_feature:
                            y = process_array_col(x, args.batch_size)
                            # print("y: ", y.shape)
                            my_dict[col] = y
                            continue
                        y = np.expand_dims(x, axis=1)
                        my_dict[col] = y
                    if random.randint(0, random_n) == 0:
                        my_dict['geek_id'][0] = [-1]
                    yield my_dict, labels

    print('生成dataset...')
    dataset_train = tf.data.Dataset.from_generator(raw_data_gen_batch_mask,
                                                   output_types=generator_output_types,
                                                   output_shapes=generator_output_shapes). \
        shuffle(buffer_size=args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_train, num_nodes, num_features, feature_names


def export_to_savedmodel(export_model, my_savedmodel_dir):
    """保存模型"""
    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    if not os.path.exists(my_savedmodel_dir):
        os.mkdir(my_savedmodel_dir)
    tf.keras.models.save_model(export_model, my_savedmodel_dir, options=options)


def load_default_embed(final_model_path, lr, embedding_dim, num_nodes, n_sampled, feature_names):
    """获取默认的geek embedding和weigth embedding"""
    tf.keras.backend.clear_session()
    tfra.dynamic_embedding.enable_inference_mode()
    export_model = build_model(
        lr=lr,
        embedding_dim=embedding_dim,
        is_training=False,
        num_nodes=num_nodes,
        n_sampled=n_sampled,
        feature_names=feature_names
    )
    export_model.load_weights(final_model_path)
    export_model.summary()

    print("Load the default geek embedding and weight embedding")
    default_inputs = {}
    for fea in feature_names:
        if fea == 'geek_id':
            default_inputs[fea] = np.array([[-1]])
        elif fea in static_array_fea_c:
            default_inputs[fea] = np.array([[-3, -3, -3]])
        else:
            default_inputs[fea] = np.array([[-3]])
    default_inputs['labels'] = np.array([[-3]])
    geek_model = tf.keras.Model(inputs=export_model.inputs,
                                outputs=export_model.get_layer("UnifiedDynamicEmbedding").output)
    geek_outputs = geek_model.predict(default_inputs)
    geek_default_embedding = geek_outputs[0, 0]

    weight_model = tf.keras.Model(inputs=export_model.inputs,
                                  outputs=export_model.get_layer("DynamicWeightEmbedding").output)
    weight_outputs = weight_model.predict(default_inputs)
    weight_default_embedding = weight_outputs
    return geek_default_embedding, weight_default_embedding


def save_final_model(online_model_path, final_model_path, lr, embedding_dim,
                     num_nodes, n_sampled, feature_names,
                     geek_default_embedding, weight_default_embedding):
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
        feature_names=feature_names,
        geek_default_initializer=list(geek_default_embedding),
        weight_default_initializer=list(weight_default_embedding)
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

    input_signatures = {}
    for fea in final_model.inputs:
        length = feature_name2len_dict[fea.name]
        code = feature_name2code_dict[fea.name]
        input_dtype = fea.dtype if fea.dtype != tf.float64 else tf.float32
        input_spec = tf.TensorSpec(shape=[None, length], dtype=input_dtype, name=str(code))
        input_signatures[fea.name] = input_spec

    @tf.function()
    def predicts(input_signatures):
        outputs = final_model(input_signatures)
        output_signatures = {}
        output_signatures['geek_vector'] = outputs
        return output_signatures

    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    signatures = predicts.get_concrete_function(input_signatures)
    # 保存到线上指定的目录
    tf.keras.models.save_model(final_model,
                               online_model_path,
                               overwrite=True,
                               include_optimizer=False,
                               save_traces=False,
                               options=options,
                               signatures=signatures)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--root_path", type=str,
                        default='')
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--max_epochs_without_improvement", type=int, default=1)
    parser.add_argument("--online_model_path", type=str, default='')

    args = parser.parse_args()
    print(f'dim:{args.embedding_dim}, batch_size:{args.batch_size}, '
          f'n_sampled:{args.n_sampled}, epochs:{args.epochs}, 当前路径:{args.root_path}, 学习率:{args.lr}, 模型最终保存路径:{args.online_model_path}, max_epochs_without_improvement:{args.max_epochs_without_improvement}')

    print("load files......")
    now_datetime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    output_folder = f'/group/recall/liuchen04/get/new/output/{now_datetime}'
    print(f'目录：用户存放训练数据的root_path:{args.root_path}')

    # 文件路径
    pair_file_names_train = get_file_names(args.root_path + 'pairs_30', ['csv'])
    side_info_file_name = get_file_names(args.root_path + 'side_info_30')[0]

    print('Generate dataset...')
    # 本地测试文件路径
    # pair_file_names_train = [""]
    # side_info_file_name = ""
    print('side_info文件:', side_info_file_name)
    print(f'pair 训练集:{pair_file_names_train}')
    dataset_train, num_nodes, num_features, feature_names = get_dataset_mask(pair_file_names_train, side_info_file_name)
    print("Build the model...")
    model = build_model(lr=args.lr,
                        embedding_dim=args.embedding_dim,
                        is_training=True,
                        num_nodes=num_nodes,
                        n_sampled=args.n_sampled,
                        feature_names=feature_names
                        )
    model.summary()
    print("Train the model...")
    log_dir = output_folder + "/log"
    print(f'日志目录:{log_dir}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100)
    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    # 不同版本参数设置不一样，tf2.6.3 patience表示多少个epoch变化小于min_delta终止训练
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=args.max_epochs_without_improvement,
                                                      min_delta=1,
                                                      verbose=1)
    callbacks_list = [tensorboard_callback, early_stopping]

    print("============")
    model.fit(
        dataset_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks_list
    )

    print("Save the model...")
    final_save_model_dir = output_folder + "/final_saved_model"
    export_to_savedmodel(model, final_save_model_dir)
    print("Finished training")

    print("Load the final model...")
    geek_default_embedding, weight_default_embedding = load_default_embed(
        final_save_model_dir, args.lr, args.embedding_dim, num_nodes, args.n_sampled, feature_names)
    print("geek_default_embedding: ", geek_default_embedding)
    print("weight_default_embedding: ", weight_default_embedding)

    print("Modify the model initializer, inputs and outputs")
    if not os.path.exists(args.online_model_path):
        os.mkdir(args.online_model_path)

    save_final_model(args.online_model_path, final_save_model_dir, args.lr, args.embedding_dim,
                     num_nodes, args.n_sampled, feature_names,
                     geek_default_embedding, weight_default_embedding)
    print("Finished save in the fixed dir")

    end_time = time.time()
    cost_time = (end_time - start_time) / 60
    print(f"Total cost {cost_time} min in train and save")

