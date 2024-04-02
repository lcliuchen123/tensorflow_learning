# coding:utf-8

# 自定义模型
# deepfm/ppnet
import pandas as pd
import tensorflow as tf
from hive_to_asena_v2 import *

# 定义模型的输入
file_path = '/group/live/liuchen04/cp/20231211/特征.csv'
f1_columns_dict, new_columns_dict, text_columns_dict, cat_columns_dict = get_features(file_path)


def process_single_feature(fea_name, fea_type, fea_cate, embedding_size, feature_bound_dict):
    """预处理单个特征"""
    if not fea_type or not fea_name:
        print("fea_type or fea_name is null")
        return

    # 根据特征类型处理特征, 数值特征/类别特征，序列特征/非序列特征
    onehot_fea = None
    embedd_fea = None

    # 文本特征没有fea_bound
    if fea_name in feature_bound_dict:
        fea_bound = feature_bound_dict[fea_name]
        # 先转化为float再转化为int，str直接转int会报错，比如int(' 0.0')
        fea_bound = [float(x) for x in fea_bound]
        if fea_type in ('int', 'long', 'int-vector', 'long-vector'):
            fea_bound = [int(x) for x in fea_bound]
        elif fea_type in ('float', 'double', 'float-vector', 'double-vector'):
            fea_bound = fea_bound
    elif fea_name in text_columns_dict:
        print("Text feature not have feature bound")
    else:
        print(f"fea_name: {fea_name}, fea_type: {fea_type} not int or float")
        return None

    #     print(f'fea_name" {fea_name}, fea_bound:{fea_bound}')
    if fea_name not in seq_feature_length_dict:
        if fea_cate == 0:
            onehot_fea = tf.feature_column.bucketized_column(
                tf.feature_column.numeric_column(key=fea_name, dtype=tf.int64), boundaries=fea_bound)
            embedd_fea = tf.feature_column.embedding_column(onehot_fea, dimension=embedding_size)
        else:
            onehot_fea = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    fea_name, fea_bound, default_value=0))
            embedd_fea = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(
                fea_name, fea_bound, default_value=0), dimension=embedding_size)
    else:
        # 文本特征等于它自身
        if fea_name in text_columns_dict:
            onehot_fea = tf.feature_column.numeric_column(fea_name, shape=(256,))
            embedd_fea = tf.feature_column.numeric_column(fea_name, shape=(256,))
        else:
            onehot_fea = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(fea_name, fea_bound, default_value=0)
            )
            embedd_fea = tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_vocabulary_list(fea_name, fea_bound, default_value=0),
                dimension=embedding_size, combiner="mean")

    return onehot_fea, embedd_fea


class sim_cp_model(tf.keras.Model):
    def __init__(self, model_type, embedding_dim, fea_num=0):
        super(sim_cp_model, self).__init__()
        self.model_type = model_type
        self.fea_num = fea_num
        self.embedding_dim = embedding_dim
        self.onehot_inputs_dict, self.dense_inputs_dict = self.process_fea()
        self.onehot_inputs = tf.keras.layers.DenseFeatures(list(self.onehot_inputs_dict.values()))
        self.dense_inputs = tf.keras.layers.DenseFeatures(list(self.dense_inputs_dict.values()))

        # layer需要定义在init函数里面，否则找不到训练变量
        # 定义deepfm模型结构
        self.dnn1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.dnn2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dnn3 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dnn_output = tf.keras.layers.Dense(units=1, activation='sigmoid')
        self.linear = tf.keras.layers.Dense(1, kernel_initializer="zeros")
        self.final_output = tf.keras.layers.Dense(units=1, activation='sigmoid')

        # 定义ppnet
        self.position_inputs = tf.keras.layers.DenseFeatures([self.dense_inputs_dict['geek_position']])
        self.other_inputs = tf.keras.layers.DenseFeatures(
            [value for key, value in self.dense_inputs_dict.items() if key != 'geek_position'])

        # gate_nn
        self.gate_nn_hidden1 = tf.keras.layers.Dense(units=1024, activation='relu')
        # 出去了geek_position, 再加上3个256维的文本特征
        self.gate_nn_output1 = tf.keras.layers.Dense(units=(self.fea_num - 4) * self.embedding_dim + 256 * 3,
                                                     activation='sigmoid')
        self.gate_nn_hidden2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.gate_nn_output2 = tf.keras.layers.Dense(units=256, activation='sigmoid')
        self.gate_nn_hidden3 = tf.keras.layers.Dense(units=256, activation='relu')
        self.gate_nn_output3 = tf.keras.layers.Dense(units=128, activation='sigmoid')

        # 开聊/达成保持一致的结构
        self.ppnet_dnn1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.ppnet_dnn2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.ppnet_dnn3 = tf.keras.layers.Dense(units=32, activation='relu')
        self.ppnet_output = tf.keras.layers.Dense(units=1, activation='sigmoid')

        # 达成gate_nn
        self.succ_gate_nn_hidden1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.succ_gate_nn_output1 = tf.keras.layers.Dense(units=(self.fea_num - 4) * self.embedding_dim + 256 * 3,
                                                          activation='sigmoid')
        self.succ_gate_nn_hidden2 = tf.keras.layers.Dense(units=512, activation='relu')
        self.succ_gate_nn_output2 = tf.keras.layers.Dense(units=256, activation='sigmoid')
        self.succ_gate_nn_hidden3 = tf.keras.layers.Dense(units=256, activation='relu')
        self.succ_gate_nn_output3 = tf.keras.layers.Dense(units=128, activation='sigmoid')

        # 达成结构
        self.succ_ppnet_dnn1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.succ_ppnet_dnn2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.succ_ppnet_dnn3 = tf.keras.layers.Dense(units=32, activation='relu')
        self.succ_ppnet_output = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def get_fea_threshold(self):
        # 获取特征的取值 读取文件获取每个特征的类别值或分位点
        print("enter the function")
        feature_bound_dict = {}
        fea_path = "/group/live/liuchen04/cp/20231211/threshold_new.csv"
        data = pd.read_csv(fea_path)
        length = len(data)
        for i in range(length):
            row = data.iloc[i]
            feature_name = row["featureName"]
            value = str(row["thresholdOrUniqueValue"]).replace('[', '').replace(']', '').replace(' ', '')

            # 如果为空就继续执行循环
            if not feature_name or not value:
                continue
            value = value.split(',')
            feature_bound_dict[feature_name] = value
        print("finish the get_fea_threshold")
        return feature_bound_dict

    def process_fea(self):
        """特征预处理: 分桶/onehot"""
        # 获取所有的特征
        all_fea_dict = {}
        file_path = '/group/live/liuchen04/cp/20231211/特征.csv'
        f1_columns_dict, new_columns_dict, text_columns_dict, cat_columns_dict = get_features(file_path)
        all_fea_dict.update(f1_columns_dict)
        all_fea_dict.update(new_columns_dict)
        all_fea_dict.update(text_columns_dict)

        # 删除的特征 job_unfit_position_recent30拒绝类特征
        #          'five_job_key_words_embed', 'five_pos_embed', 'five_job_title_embed'
        deleted_features = ['job_addf_revage_recent30', 'job_addf_gender_recent30',
                            'five_addf_gender_8_14days_gof', 'five_job_suc_gender_recent10']

        fea_bound_dict = self.get_fea_threshold()
        fea_num = 0
        onehot_fea_dict = {}
        dense_fea_dict = {}
        for fea_name in all_fea_dict:
            # 删除拒绝类特征，观察模型效果
            #             if 'refuse' in fea_name or fea_name == 'job_unfit_position_recent30':
            #                 continue

            if fea_name not in deleted_features:
                fea_type = all_fea_dict[fea_name]
                fea_cate = cat_columns_dict[fea_name]
                onehot_fea, embedd_fea = process_single_feature(fea_name, fea_type, fea_cate,
                                                                self.embedding_dim,
                                                                fea_bound_dict)
                onehot_fea_dict[fea_name] = onehot_fea
                dense_fea_dict[fea_name] = embedd_fea
                fea_num += 1
        self.fea_num = fea_num
        print("fea_num: ", fea_num)
        return onehot_fea_dict, dense_fea_dict

    def deepfm(self, inputs):
        """单目标开聊"""
        # deep层
        #         print("inputs_1: ", inputs)
        dnn0 = self.dense_inputs(inputs)
        #         print("dnn0: ", dnn0)
        dnn1 = self.dnn1(dnn0)
        #         print("dnn1: ", dnn1)
        dnn2 = self.dnn2(dnn1)
        #         print("dnn2: ", dnn2)
        dnn3 = self.dnn3(dnn2)
        #         print("dnn3: ", dnn3)
        dnn_output = self.dnn_output(dnn3)
        #         print("dnn_output: ", dnn_output)

        # 一阶项
        onehot_inputs = self.onehot_inputs(inputs)
        first_order = self.linear(onehot_inputs)

        # 二阶项
        embedd = tf.reshape(self.dense_inputs(inputs), (-1, self.fea_num, self.embedding_dim))
        sum_square = tf.square(tf.reduce_sum(embedd, axis=1))
        square_sum = tf.reduce_sum(tf.square(embedd), axis=1)
        second_order = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True)

        # 输出为字典格式
        final_output = {}
        final_output['is_addf'] = self.final_output(first_order + second_order + dnn_output)
        return final_output

    def ppnet(self, inputs):
        """开聊和达成目标"""
        outputs = {}
        # 构造输入
        position_inputs = self.position_inputs(inputs)
        other_inputs = self.other_inputs(inputs)

        # 构造开聊目标
        addf_inputs = tf.concat([tf.stop_gradient(other_inputs), position_inputs], axis=1)

        # 第一层
        hidden1 = self.gate_nn_hidden1(addf_inputs)
        gate_nn_output1 = 2 * self.gate_nn_output1(hidden1)
        addf_input1 = tf.multiply(other_inputs, gate_nn_output1)
        addf_output1 = self.ppnet_dnn1(addf_input1)  # batch_size, 256

        # 第二层 gate nn的输入是一样的
        hidden2 = self.gate_nn_hidden2(addf_inputs)
        gate_nn_output2 = 2 * self.gate_nn_output2(hidden2)  # batch_size, 256
        addf_input2 = tf.multiply(addf_output1, gate_nn_output2)
        addf_output2 = self.ppnet_dnn2(addf_input2)  # batch_size, 128

        # 第三层 gate nn的输入是一样的
        hidden3 = self.gate_nn_hidden3(addf_inputs)
        gate_nn_output3 = 2 * self.gate_nn_output3(hidden3)  # batch_size, 128
        addf_input3 = tf.multiply(addf_output2, gate_nn_output3)
        addf_output3 = self.ppnet_dnn3(addf_input3)  # batch_size, 32

        # 开聊目标输出
        addf_output = self.ppnet_output(addf_output3)
        outputs['is_addf'] = addf_output

        # 构造达成目标
        succ_inputs = tf.concat([tf.stop_gradient(other_inputs), position_inputs], axis=1)
        # 第一层
        succ_hidden1 = self.succ_gate_nn_hidden1(succ_inputs)
        succ_gate_nn_output1 = 2 * self.succ_gate_nn_output1(succ_hidden1)
        succ_input1 = tf.multiply(other_inputs, succ_gate_nn_output1)
        succ_output1 = self.succ_ppnet_dnn1(succ_input1)  # batch_size, 256

        # 第二层 gate nn的输入是一样的
        succ_hidden2 = self.succ_gate_nn_hidden2(addf_inputs)
        succ_gate_nn_output2 = 2 * self.succ_gate_nn_output2(succ_hidden2)  # batch_size, 256
        succ_input2 = tf.multiply(succ_output1, succ_gate_nn_output2)
        succ_output2 = self.succ_ppnet_dnn2(succ_input2)  # batch_size, 128

        # 第三层 gate nn的输入是一样的
        succ_hidden3 = self.succ_gate_nn_hidden3(addf_inputs)
        succ_gate_nn_output3 = 2 * self.succ_gate_nn_output3(succ_hidden3)  # batch_size, 128
        succ_input3 = tf.multiply(succ_output2, succ_gate_nn_output3)
        succ_output3 = self.succ_ppnet_dnn3(succ_input3)  # batch_size, 32

        # 达成目标输出
        succ_output = self.succ_ppnet_output(succ_output3)
        outputs['is_success'] = succ_output

        return outputs

    def call(self, inputs, training=False):
        #         print("inputs: ", inputs)
        outputs = None
        if self.model_type == 'deepfm':
            outputs = self.deepfm(inputs)
        elif self.model_type == 'ppnet':
            outputs = self.ppnet(inputs)
        else:
            print("Input the model_type in deepfm or ppnet")
        return outputs


