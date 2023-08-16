# coding:utf-8

# 参考 https://github.com/StephenBo-China/DIEN-DIN/blob/main/model.py
import tensorflow as tf
import keras.backend as K


class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_units_list=(80, 40, 1), attention_activation='sigmoid', support_mask=True):
        super(Attention, self).__init__()
        self.hidden_units_list = hidden_units_list
        self.attention_activation = attention_activation
        self.support_mask = support_mask

    def build(self, input_shape):
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        """
        :param x:
              x[0]: target_emb [batch_size, hidden_units],
              x[1]: click_emb [batch_size, max_len, hidden_units],
              x[2]: click_length  [batch_size] 用户行为序列真实长度
        :param mask:
        """
        assert len(x) == 3
        target_emb, click_emb, click_length = x[0], x[1], x[2]
        hidden_units = K.int_shape(click_emb)[-1]
        max_len = tf.shape(click_emb)[1]

        target_emb = tf.tile(target_emb, [1, max_len])
        target_emb = tf.reshape(target_emb, [-1, max_len, hidden_units])  # [batch_size, max_len, hidden_units]
        concat_input = K.concatenate([click_emb, target_emb, click_emb - target_emb, click_emb * target_emb], axis=2)

        # 2层全连接层 + 1层线性层
        for i in range(len(self.hidden_units_list)):
            if i == (len(self.hidden_units_list) - 1):
                activation = None
            else:
                activation = self.attention_activation
            output = tf.keras.layers.Dense(self.hidden_units_list[i], activation=activation)(concat_input)
            concat_input = output
        output = tf.reshape(output, [-1, 1, max_len])  # [batch_size, 1, max_len]

        # 用户行为序列不足的补到最大长度，权重设为一个较小的数字
        if self.support_mask:
            mask = tf.sequence_mask(click_length, max_len)
            padding = tf.ones_like(output) * (-1e12)
            output = tf.where(mask, output, padding)

        # 进行缩放
        output = output / (hidden_units ** 0.5)
        output = K.softmax(output)

        # sum pooling
        output = tf.matmul(output, click_emb)  # [batch_size, 1, hidden_units]
        output = tf.squeeze(output)  # [batch_size, hidden_units]
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


def get_final_loss(y_true, y_pred):
    # 更改label
    labels = tf.one_hot(y_true, 2)

    # 定义loss
    final_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=tf.cast(labels, dtype=tf.float32)))
    return final_loss


def build_din_model(lr, num_items, emb_dim, is_attention, num_shops, num_cates):
    """构造din模型"""
    # 输入层
    feature_type = {'click_nums_2d': tf.int64, 'click_rate_2d': tf.float32,
                    'age': tf.int64, 'gender': tf.int64,
                    'click_id_recent_10': tf.int64, 'click_shop_id_recent10': tf.int64,
                    'click_cate_id_recent_10': tf.int64
                    }
    feature_length = {'click_nums_2d': 1, 'click_rate_2d': 1,
                      'age': 1, 'gender': 1, 'click_id_recent_10': 10,
                      'click_shop_id_recent10': 10,
                      'click_cate_id_recent_10': 10
                    }
    boundary_dict = {'age': [10, 20, 30, 60], 'click_nums_2d': [1, 2, 3, 4, 5], 'click_rate_2d': [0.1, 0.2, 0.3, 0.4, 0.5]}
    vocab_dict = {'gender': [0, 1]}
    num_features = ['age', 'click_nums_2d', 'click_rate_2d']
    cate_features = ['gender']
    seq_features = ['click_id_recent_10', 'click_shop_id_recent10', 'click_cate_id_recent_10']

    all_features = num_features + cate_features + seq_features
    input_dict = {}
    for fea in all_features:
        length = feature_length[fea]
        dtype = feature_type[fea]
        input_dict[fea] = tf.keras.layers.Input(shape=(length,), dtype=dtype, name=fea)

    input_dict['target_id'] = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='target_id')
    input_dict['target_cate_id'] = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='target_cate_id')
    input_dict['target_shop_id'] = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='target_shop_id')
    input_dict['click_length'] = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='click_length')
    input_dict['label'] = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='label')

    # embedding层 feature_columns
    one_hot_dict = {}
    emb_dict = {}
    for fea in all_features:
        if fea in num_features:
            num_fea = tf.feature_column.numeric_column(key=fea, dtype=feature_type[fea])
            one_hot_dict[fea] = tf.feature_column.bucketized_column(num_fea, boundaries=boundary_dict[fea])
            emb_dict[fea] = tf.feature_column.embedding_column(one_hot_dict[fea], dimension=32)
        elif fea in cate_features:
            cate_fea = tf.feature_column.categorical_column_with_vocabulary_list(key=fea, vocabulary_list=vocab_dict[fea])
            one_hot_dict[fea] = tf.feature_column.indicator_column(cate_fea)
            emb_dict[fea] = tf.feature_column.embedding_column(cate_fea, dimension=32)
        elif fea in seq_features:
            # 如果序列特征聚合，不做attention可以直接利用下面的代码
            if not is_attention:
                seq_cate_fea = tf.feature_column.categorical_column_with_vocabulary_list(key=fea, vocabulary_list=vocab_dict[fea])
                one_hot_dict[fea] = tf.feature_column.indicator_column(seq_cate_fea)
                emb_dict[fea] = tf.feature_column.embedding_column(seq_cate_fea, dimension=32, combiner='mean')
            else:
                print("用户行为序列特征需要特殊处理")
        else:
            print("feature not in num_features, cate_features and seq_cate_features")

    # 将user侧和item侧特征拼凑在一起
    fea_list = list(emb_dict.keys())
    print("fea_list: ", fea_list)
    fea_value_list = list(emb_dict.values())
    print(fea_value_list)
    user_item_emb_fea = tf.keras.layers.DenseFeatures(fea_value_list)(input_dict)

    # 处理用户行为序列特征 click_id_recent_10, click_shop_id_recent10, click_cate_id_recent_10
    item_id_embedding = tf.keras.layers.Embedding(
        input_dim=num_items,
        output_dim=emb_dim,
        embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=1234))
    shop_id_embedding = tf.keras.layers.Embedding(
        input_dim=num_shops,
        output_dim=emb_dim,
        embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=1234))

    cate_id_embedding = tf.keras.layers.Embedding(
        input_dim=num_cates,
        output_dim=emb_dim,
        embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=1234))

    # target_id_embedding
    target_id_emb = item_id_embedding(input_dict['target_id'])   # [batch_size, 1, output_dim]
    target_cate_emb = item_id_embedding(input_dict['target_cate_id'])
    target_shop_emb = item_id_embedding(input_dict['target_shop_id'])
    target_emb = [target_id_emb, target_shop_emb, target_cate_emb]
    target_seq_emb = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1], x[2]], axis=-1))(target_emb)  # [batch_size, 1, 3 * output_dim]
    target_seq_emb = tf.keras.layers.Lambda(lambda x: K.squeeze(x, axis=1))(target_seq_emb)  # [batch_size, 3*output_dim]

    # click_id_embedding
    item_id_emb = item_id_embedding(input_dict['click_id_recent_10'])
    shop_id_emb = shop_id_embedding(input_dict['click_shop_id_recent10'])
    cate_id_emb = cate_id_embedding(input_dict['click_cate_id_recent_10'])
    click_id_emb = [item_id_emb, shop_id_emb, cate_id_emb]
    click_seq_emb = tf.keras.layers.Lambda(lambda x: K.concatenate([x[0], x[1], x[2]], axis=-1))(click_id_emb)

    # 构建点击序列和目标id的关系 Attention层
    din_attention = Attention()([target_seq_emb, click_seq_emb, input_dict['click_length']])
    din_attention = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 3 * emb_dim]))(din_attention)

    # concat embedding 拼接embedding层
    print(user_item_emb_fea.shape)
    print(din_attention.shape)
    print(target_seq_emb.shape)
    join_emb = tf.concat([user_item_emb_fea, din_attention, target_seq_emb], -1)
    print(join_emb.shape)

    # 再经过两层到输出
    fc1 = tf.keras.layers.Dense(200, activation="relu")(join_emb)
    print(fc1.shape)
    fc2 = tf.keras.layers.Dense(80, activation="relu")(fc1)
    print(fc2.shape)
    logit = tf.keras.layers.Dense(2, activation=None)(fc2)
    print(logit.shape)
    outputs = tf.keras.activations.softmax(logit)  # [batch_size, 2]

    # 定义模型
    model = tf.keras.Model(inputs=input_dict, outputs=outputs)
    model.summary()

    # 定义学习器
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='get_final_loss', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    lr = 0.00001
    num_items = 10000
    emb_dim = 32
    is_attention = True
    num_shops = 1000
    num_cates = 100
    model = build_din_model(lr, num_items, emb_dim, is_attention, num_shops, num_cates)
