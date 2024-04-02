# coding:utf-8

# 保存各种超参数配置，比如模型使用的特征,标签的分段值

from ..feature import feature_conf
import datetime as dt


model_feature_settings = {
    'v1': {'select_features': ['fea1', 'fea2']},
    'v2': {'select_features': ['fea3', 'fea4']}
}

model_conf = {
    'numeric_features': [],
    'categorical_features': []
}


def get_model_conf(version):
    """
    根据输入版本返回对应配置
    :param version: 模型版本
    :return: 模型配置
    """
    # 默认版本
    if version is None:
        version = 'v2'

    # 处理模型配置
    model_conf['numeric_features'] = [f for f in model_conf['numeric_features'] if f in model_feature_settings[version]['selected_features']]
    model_conf['categorical_features'] = [f for f in model_conf['categorical_features'] if f in model_feature_settings[version]['selected_features']]

    return model_conf
