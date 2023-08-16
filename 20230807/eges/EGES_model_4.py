# -*- coding: utf-8 -*-

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from feature_conf import *
from embedding_gpu import *
from bucketize import *
from neg_sample import *


def build_input_embedding(fea_name_list, inputs_dict, embedding_dim, initializer):
    fea_name_list = list(fea_name_list)
    # print("fea_name_list: ", fea_name_list)
    # 获取每个特征的索引和维度
    fea_code_dic = {}
    fea_dim_dic = {}
    for i, name in enumerate(fea_name_list):
        fea_code_dic[name] = i
        if name in static_array_fea_c:
            fea_dim_dic[name] = 3
        else:
            fea_dim_dic[name] = 1

    # tfra处理序列和非序列特征
    fea_id_tensors = list()
    fea_id_split_dims = list()
    fea_id_is_sequence_feature = list()
    for fea_name, fea_tensor in inputs_dict.items():
        if fea_name not in fea_name_list:
            # print("fea_name: ", fea_name)
            continue
        fea_tensor = fea_tensor if fea_tensor.dtype == tf.int64 else tf.cast(fea_tensor, tf.int64)
        id_tensor_prefix_code = int(fea_code_dic[fea_name]) << 47
        id_tensor = tf.add(fea_tensor, id_tensor_prefix_code)
        fea_id_tensors.append(id_tensor)
        dim = fea_dim_dic[fea_name]
        fea_id_split_dims.append(dim)
        if dim > 1:
            fea_id_is_sequence_feature.append(True)
        else:
            fea_id_is_sequence_feature.append(False)

    # print("fea_id_is_sequence_feature: ", fea_id_is_sequence_feature)
    # print("fea_id_tensors: ", fea_id_tensors)
    fea_id_tensors_concat = tf.keras.layers.Concatenate(axis=1)(fea_id_tensors)
    # print("fea_id_tensors_concat: ", fea_id_tensors_concat)

    # tfra cpu / gpu都可以，主要是为了存储
    gpu_device = "/job:localhost/replica:0/task:0/CPU:0"
    # print("embedding_dim: ", embedding_dim)
    fea_embedding_out_concat = EmbeddingLayerGPU(
        embedding_size=embedding_dim,
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        initializer=initializer,
        devices=gpu_device,
        name="UnifiedDynamicEmbedding",
        init_capacity=1000000,
        kv_creator=None)(fea_id_tensors_concat)

    # 解析embedding
    embedding_out = list()
    embedding_outs = list()
    # (feature_combin_num, (batch, dim, emb_size))
    embedding_out.extend(tf.split(fea_embedding_out_concat, fea_id_split_dims, axis=1))
    assert ((len(fea_id_is_sequence_feature)) == len(embedding_out))
    for i, embedding in enumerate(embedding_out):
        if fea_id_is_sequence_feature[i]:
            # (feature_combin_num, (batch, x, emb_size))
            embedding_vec = tf.math.reduce_mean(embedding, axis=1, keepdims=True)
        else:
            embedding_vec = embedding
        embedding_outs.append(embedding_vec)

    # 把特征向量拼接到一起
    # print("embedding_outs: ", embedding_outs)  # [batch_size, 1, dim]
    stack_embed = tf.stack(embedding_outs, axis=1)
    stack_embed = tf.reshape(stack_embed, [-1, len(fea_name_list), embedding_dim])
    # print("stack_embed: ", stack_embed)
    return stack_embed


def basic_loss_function(y_true, y_pred):
    """https://zhuanlan.zhihu.com/p/151057845"""
    return tf.math.reduce_mean(y_pred)


def build_model(lr, embedding_dim, is_training, num_nodes, n_sampled, feature_names,
                geek_default_initializer=None, weight_default_initializer=None):
    """构建模型"""
    if is_training:
        geek_default_initializer = tf.keras.initializers.VarianceScaling()
        weight_default_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=123)
    else:
        if not geek_default_initializer:
            geek_default_initializer = geek_default_initializer
        else:
            geek_default_initializer = tf.keras.initializers.Zeros()

        if not weight_default_initializer:
            weight_default_initializer = weight_default_initializer
        else:
            weight_default_initializer = tf.keras.initializers.Zeros()

    # step1 构造输入
    ori_inputs_dict = {}
    for fea_name in feature_names_c:
        if fea_name in static_array_fea_c:
            ori_inputs_dict[fea_name] = tf.keras.layers.Input(shape=(3,), dtype=tf.int64, name=fea_name)
        else:
            ori_inputs_dict[fea_name] = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name=fea_name)
    # 添加label列
    ori_inputs_dict['labels'] = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='labels')
    # print("===ori_inputs_dict: ===\n", ori_inputs_dict)

    # step2 连续特征进行分段  bucekt层
    inputs_dict = {}
    for lianxu_feature in static_numeric_dict_c.keys():
        boundaries = static_numeric_dict_c[lianxu_feature]
        inputs_dict[lianxu_feature] = tf.cast(Bucketize(boundaries)(ori_inputs_dict[lianxu_feature]),
                                                       dtype=tf.int64)
        for fea_name in feature_names:
            if fea_name not in static_numeric_dict_c.keys():
                inputs_dict[fea_name] = ori_inputs_dict[fea_name]
    # print("===inputs_dict: ===\n", inputs_dict)

    # step3 embedding层
    stack_embed = build_input_embedding(feature_names, inputs_dict, embedding_dim, geek_default_initializer)
    # print("===stack_embed: ===\n", stack_embed)

    # tfra定义权重
    gpu_device = "/job:localhost/replica:0/task:0/CPU:0"
    alpha_embedding = EmbeddingLayerGPU(
        embedding_size=len(feature_names),
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        initializer=weight_default_initializer,
        devices=gpu_device,
        name="DynamicWeightEmbedding",
        init_capacity=1000000 * 4,
        kv_creator=None)(ori_inputs_dict['geek_id'])

    # (batch_size, 1, feat_nums)
    alpha_embeds = tf.math.exp(alpha_embedding)
    alpha_sum = tf.reduce_sum(alpha_embeds, axis=-1)
    # (batch_size, 1, embed_size)
    merge_embeds = tf.matmul(alpha_embeds, stack_embed)
    # (batch_size, embed_size), 归一化
    merge_embeds = tf.squeeze(merge_embeds, axis=1) / alpha_sum

    # print("===att_embeds: ===\n", att_embeds)

    # 负采样loss
    cost = SampledSoftmax(num_nodes,
                          n_sampled,
                          l2_reg=0.00001,
                          seed=1234)([merge_embeds, ori_inputs_dict["labels"]])
    # print("===cost: ===\n", cost)

    model = tf.keras.Model(inputs=ori_inputs_dict, outputs=cost)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=False)
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=basic_loss_function,
        metrics=None)
    return model




