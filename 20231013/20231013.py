# coding:utf-8

# 复现dien模型

import tensorflow as tf
import numpy as np
import keras.backend as K


class Attention(tf.keras.layers):
    def __init__(self, support_mask=True):
        super(Attention, self).__init__()
        self.support_mask = support_mask

    def build(self, input_shape):
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        """
          x[0]: target_id_embed [batch_size, 1, dim]
          x[1]: hidden_embed [batch_size, max_seq_length, dim]
          x[2]: real_length [batch_size, 1] 真实的长度
        """
        assert len(x) == 3

        target_id_emb, hidden_emb, real_length = x
        max_seq_length = tf.shape(hidden_emb)[1]
        dim_target = tf.shape(target_id_emb)[-1]

        output = tf.keras.layers.Dense(units=dim_target, activation='relu')(hidden_emb)  # [b, s, d_t]
        output_with_target = tf.matmul(output, tf.transpose(target_id_emb))  # [b, max_seq_length, 1]
        output_with_target = tf.reshape(output_with_target, [-1, 1, max_seq_length])

        if self.support_mask:
            # 返回一个mask的tensor, true or false
            mask = tf.sequence_mask(real_length, max_seq_length)  # [batch_size, 1, max_seq_length]
            # 填补的部分乘以一个比较小的数字，权重为0
            padding = tf.ones_like(output_with_target) * 1e-12
            output_with_target = tf.where(mask, output_with_target, padding)

        weights = K.softmax(output_with_target)  # [batch_size, 1, max_seq_length]
        weights = tf.squeeze(weights)  # [batch_size, max_seq_length]
        return weights

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


def build_model(batch_size, lr, embedding_dim, fc_units=[200, 80, 2]):
    inputs = {}
    # 直接定义代码比较长
    # # 用户侧特征
    # inputs['user_age'] = tf.keras.Input(shape=(1,), dtype=tf.float64, name='user_age')
    # inputs['user_gender'] = tf.keras.Input(shape=(1,), dtype=tf.int64, name='user_gender')
    #
    # # 上下文特征
    # inputs['sys'] = tf.keras.Input(shape=(1, ), dtype=tf.int64, name='sys')
    # inputs['week'] = tf.keras.Input(shape=(1, ), dtype=tf.int64, name='week')
    #
    # # 候选item特征
    # inputs['item_click_rate'] = tf.keras.Input(shape=(1,), dtype=tf.float64, name='item_click_rate')
    # inputs['item_list_num'] = tf.keras.Input(shape=(1,), dtype=tf.int64, name='item_list_num')
    # inputs['item_id'] = tf.keras.Input(shape=(1,), dtype=tf.int64, name='item_id')
    #
    # # 行为序列特征
    # # 用户行为序列长度可能不一致, 都填充到最大长度，然后用一个字段保留其真实长度
    # inputs['user_behavior_seq'] = tf.keras.Input(shape=(30, ), dtype=tf.int64, name='user_behavior_seq')
    # inputs['user_behavior_seq_length'] = tf.keras.Input(shape=(1,), dtype=tf.int64, name='user_behavior_seq_length')

    # 按照特征类型分类 数值/类别/序列, 比上面的方式简洁
    feature_type = {'user_age': tf.float64, 'user_gender':tf.int64, 'sys': tf.int64, 'week':tf.int64,
                    'item_click_rate': tf.float64, 'item_list_num': tf.int64, 'item_id':tf.int64, 'label': tf.int64,
                    'user_behavior_seq': tf.int64, 'user_behavior_seq_length': tf.int64, 'user_seq':tf.int64}
    feature_length = {'user_age': 1, 'user_gender': 1, 'sys': 1, 'week': 1, 'user_seq': 30,
                      'item_click_rate': 1, 'item_list_num': 1, 'item_id': 1,
                      'user_behavior_seq': 30, 'user_behavior_seq_length': 1, 'label': 1}

    num_features = ['user_age', 'item_list_num', 'item_click_rate', 'user_behavior_seq_length']
    cat_features = ['user_gender', 'sys', 'week']
    seq_features = ['user_seq']
    id_features = ['item_id', 'user_behavior_seq']
    all_features = num_features + cat_features + seq_features + id_features
    for feature in all_features:
        dtype = feature_type[feature]
        length = feature_length[feature]
        inputs[feature] = tf.keras.Input(shape=(length,), dtype=dtype, name=feature)
    print("inputs: ", inputs)

    # embedding层
    one_hot_dict = {}
    embedd_dict = {}
    for feature in all_features:
        if feature in num_features:
            numer_fea = tf.feature_column.numeric_column(key=feature, dtype=feature_type[feature])
            one_hot_dict[feature] = tf.feature_column.bucketized_column(numer_fea, boundaries=[1, 2, 3])
            embedd_dict[feature] = tf.feature_column.embedding_column(one_hot_dict[feature], dimension=32)
        elif feature in cat_features:
            cat_fea = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=[1, 2, 3])
            one_hot_dict[feature] = tf.feature_column.indicator_column(cat_fea)
            embedd_dict[feature] = tf.feature_column.embedding_column(one_hot_dict[feature], dimension=32)
        elif feature in seq_features:
            seq_fea = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=[1, 2, 3])
            one_hot_dict[feature] = tf.feature_column.indicator_column(seq_fea)
            embedd_dict[feature] = tf.feature_column.embedding_column(one_hot_dict[feature], dimension=32, combiner='sum')
        elif feature in id_features:
            print("id 特征需要特殊处理")
    # 合并部分特征的embedding
    part_fea = embedd_dict.values()
    part_features = tf.keras.layers.DenseFeatures(part_fea)(inputs)

    num_items = 100000
    item_id_embedding = tf.keras.layers.Embedding(input_dim=num_items,
                                                  output_dim=embedding_dim,
                                                  embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1e-4, seed=1234))
    embedd_dict['item_id'] = item_id_embedding(inputs['item_id'])
    embedd_dict['user_behavior_seq'] = item_id_embedding(inputs['user_behavior_seq'])

    # 兴趣变化提取层
    lstm = tf.keras.layers.GRU(32, return_sequences=True, return_state=True)
    whole_seq_output, _, _ = lstm(embedd_dict['user_behavior_seq'])

    # 辅助loss  不符合预期
    true_labels = inputs['user_behavior_seq']  # item_id序列
    # biases定义方式不确定对错，待办
    biases = tf.keras.initializers.RandomNormal(mean=0, stddev=1e-4, seed=123)
    if tf.keras.backend.learning_phase():
        first_loss = tf.nn.sampled_softmax_loss(
            weights=item_id_embedding,
            biases=biases,
            labels=true_labels,
            inputs=whole_seq_output)
    else:
        logits = tf.matmul(whole_seq_output, tf.transpose(item_id_embedding))
        logits = tf.nn.bias_add(logits, biases)
        labels_one_hot = tf.one_hot(true_labels, num_items)
        first_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits)
    # todo
    # 兴趣变化进化层 AUGRU GRU的更新门权重乘以attention值  batch_size, dim
    # u_t^ = a * u_t, h_t^ = (1 - u_t^) * h_t-1^ + u_t^ * h_t^^
    hidden_output = None  # batch_size, dim

    # 合并所有的embedd
    target_id = tf.reshape(embedd_dict['item_id'], [-1, embedding_dim])
    emb_inputs = tf.concat([hidden_output, target_id, part_features])
    emb_outputs = None
    for i in range(len(fc_units)):
        activation = None
        if i < 2:
            activation = 'relu'
        emb_outputs = tf.keras.layers.Dense(units=fc_units[i], activation=activation)(emb_inputs)
        emb_inputs = emb_outputs

    final_output = tf.keras.activations.softmax(emb_outputs)  # [batch_size, 2]
    # 想获取最大的类别
    # max_class = tf.reduce_max(final_output, axis=1)

    model = tf.keras.Model(inputs=inputs, outputs=final_output)
    model.summary()

    def final_loss(y_pred, labels):
        labels = tf.one_hot(labels, 2)
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_pred, labels=tf.cast(labels, dtype=tf.float32)))
        final_loss = first_loss + loss
        return final_loss

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=final_loss,
                  metrics=['accuracy'])
    return model






















