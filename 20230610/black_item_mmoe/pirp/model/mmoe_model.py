# coding:utf-8

from .base_model import BaseModel
from .layer import MMoELayer, TowerLayer
import tensorflow as tf


class MMoEModel(BaseModel):

    def __init__(self, conf, version=None):
        super(MMoEModel, self).__init__(conf, version)
        # task数量
        self.task_num = conf['task_num']
        # MMoE的expert数量
        self.expert_num = conf['expert_num']
        # MMoE的expert网络units，列表为从底层到顶层每层的输出unit
        self.expert_units = conf['expert_units']
        # Tower网络units
        self.tower_units = conf['tower_units']

    def build(self):
        # build inputs
        self.build_inputs()
        self.build_dense_input_layer()

        # feature layer
        feature_layer = tf.keras.layers.DenseFeatures(self.dense_feature_columns, name='feature_layer')(self.inputs)
        # MMoE layer
        mmoe_layer = MMoELayer(units=self.expert_units, num_experts=self.expert_num, num_tasks=self.task_num, name='mmoe_layer')(feature_layer)
        # outputs
        outputs = {}
        for idx, target in enumerate(self.targets):
            outputs[target] = TowerLayer(target, self.tower_units + [self.targets[target]['bin_cnt']], output_activation='sigmoid')(mmoe_layer[idx])

        self.model = tf.keras.Model(inputs=self.inputs, outputs=outputs, name='pirp_mmoe_or')
