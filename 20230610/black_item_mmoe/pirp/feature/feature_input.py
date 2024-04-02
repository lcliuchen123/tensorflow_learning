# coding:utf-8

import tensorflow as tf
from .feature_conf import feature_conf
from ..utils.helper import get_dtype


class FeatureInput:
    @staticmethod
    def create_input(name):
        feature_type = feature_conf[name]['type']
        dtype = get_dtype(feature_conf[name]['data_type'])
        if feature_type in ['id', 'categorical']:
            return tf.keras.Input(shape=(feature_conf[name].get('shape', 1),), name=name, dtype=dtype)
        elif feature_type == 'numeric':
            return tf.keras.Input(shape=(feature_conf[name].get('shape', 1),), name=name, dtype=dtype)

    @staticmethod
    def create_feature(name, discrete=True, use_embedding=True, embedding_dim=16):
        feature_type = feature_conf[name]['type']
        dtype = get_dtype(feature_conf[name]['data_type'])
        if feature_type == 'id':
            categorical_col = tf.feature_column.categorical_column_with_hash_bucket(name, feature_conf[name]['buckets'], dtype=dtype)
            if use_embedding:
                return tf.feature_column.embedding_column(categorical_col, dimension=embedding_dim)
            else:
                return tf.feature_column.indicator_column(categorical_col)
        elif feature_type == 'numeric':
            numeric_col = tf.feature_column.numeric_column(name, shape=(feature_conf[name].get('shape', 1),), dtype=dtype)
            if discrete:
                bucket_col = tf.feature_column.bucketized_column(numeric_col, boundaries=feature_conf[name]['boundaries'])
                if use_embedding:
                    return tf.feature_column.embedding_column(bucket_col, dimension=embedding_dim)
                else:
                    return bucket_col
            else:
                return numeric_col
        elif feature_type == 'categorical':
            categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(name, vocabulary_list=feature_conf[name]['vocabulary_list'], dtype=dtype, default_value=0)
            if use_embedding:
                return tf.feature_column.embedding_column(categorical_col, dimension=embedding_dim)
            else:
                return tf.feature_column.indicator_column(categorical_col)

