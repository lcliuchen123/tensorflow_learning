# coding:utf-8

from abc import abstractmethod
from ..feature import FeatureInput, feature_conf
from ..utils import get_dtype
from ..eval import EvalMetric
from .loss import OrdinalCrossEntropy
from .metric import ClassAUC
from .layer import LookupLayer

import tensorflow as tf
import datetime as dt
import os

feature_name2code_dict = {'fea_name': 'fea_code'}


class BaseModel:

    def __init__(self, conf, version=None):
        # 模型
        self.model = None
        # embedding dimension
        self.embedding_dim = conf['embedding_dim']
        # 多层全连接网络每层units
        self.tower_units = conf['tower_units']

        # 目标
        self.targets = conf['targets']

        # 路径
        ds = dt.date.today().isoformat()
        self.current_model_path = conf['current_model_path']
        if version is None:
            self.history_model_path = os.path.join(conf['history_model_path'], ds)
            self.log_path = os.path.join(conf['log_path'], ds)
        else:
            self.history_model_path = os.path.join(conf['history_model_path'], ds, version)
            self.log_path = os.path.join(conf['log_path'], ds, version)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # 各类型特征
        self.features = conf['numeric_features'] + conf['categorical_features']

        # inputs, dict of tf.keras.Input
        self.inputs = {}
        # embedding feature columns
        self.dense_feature_columns = []
        # indicator feature columns
        self.sparse_feature_columns = []
        # self defined
        self.feature_columns = []

        # 基础模型的输出阈值
        self.thresholds = None

    def build_inputs(self):
        for name in self.features:
            self.inputs[name] = FeatureInput.create_input(name)

    def build_dense_input_layer(self):
        for name in self.features:
            self.dense_feature_columns.append(
                FeatureInput.create_feature(name, use_embedding=True, embedding_dim=self.embedding_dim))

    def build_sparse_input_layer(self):
        for name in self.features:
            self.sparse_feature_columns.append(FeatureInput.create_feature(name, use_embedding=False))

    @abstractmethod
    def build(self):
        pass

    def compile(self):
        # metrics, losses, loss_weights
        metrics, losses, loss_weights = {}, {}, {}
        for target in self.targets:
            metrics[target] = ClassAUC(target, self.targets[target], name='auc')
            losses[target] = OrdinalCrossEntropy(self.targets[target], name='oce')
            loss_weights[target] = self.targets[target]['target_weight']
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=metrics, loss=losses,
                           loss_weights=loss_weights)

    def fit(self, dataset, epochs=1, verbose=1, validation_split=0.0, validation_data=None, shuffle=True,
            class_weight=None, steps_per_epoch=None, validation_steps=None, max_queue_size=10, workers=1,
            use_multiprocessing=False, callbacks=None):
        self.model.fit(dataset,
                       epochs=epochs,
                       verbose=verbose,
                       validation_split=validation_split,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       class_weight=class_weight,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       max_queue_size=max_queue_size,
                       workers=workers,
                       use_multiprocessing=use_multiprocessing,
                       callbacks=callbacks)

    def predict(self, dataset, steps=None, verbose=1):
        return self.model.predict(dataset, steps=steps, verbose=verbose)

    def evaluate(self, dataset, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False):
        self.model.evaluate(dataset,
                            batch_size=batch_size,
                            verbose=verbose,
                            sample_weight=sample_weight,
                            steps=steps,
                            callbacks=callbacks,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            return_dict=return_dict)

    def evaluate_all(self, dataset):
        eval_metric = EvalMetric(self.model, dataset, self.targets, self.log_path)
        # 计算auc和保存相关评估数据，计算阈值
        self.thresholds = eval_metric.eval_thresholds()

    def package_model(self):
        """
        封装基础模型，根据评估的阈值直接输出目标的预测档位下限值
        更新self.model为封装后的模型
        :return: None
        """
        # 新输出
        outputs = {}

        # 对每一个目标
        #         for target in self.targets:
        #             bins = self.targets[target]['bins']
        #             # 取最大的label
        #             target_pred_labels = tf.cast(tf.greater_equal(self.model(self.inputs)[target], self.thresholds[target]), tf.int32)
        #             target_pred = tf.multiply(target_pred_labels, tf.range(1, len(bins)+1))
        #             target_pred = tf.cast(tf.reduce_max(target_pred, axis=1), dtype=tf.int32)
        #             outputs[target] = LookupLayer('{0}_lb'.format(target), bins)(target_pred)

        for target in self.targets:
            bins = self.targets[target]['bins']
            # 取sum
            target_pred_labels = tf.cast(tf.greater_equal(self.model(self.inputs)[target], self.thresholds[target]),
                                         tf.int32)
            target_pred = tf.reduce_sum(target_pred_labels, axis=1)
            outputs[target] = LookupLayer('{0}_lb'.format(target), bins)(target_pred)

        # model
        self.model = tf.keras.Model(inputs=self.inputs, outputs=outputs)

    def save(self):
        print("Enter the save function")
        # feature_code未确定，暂时先用feature_name代替
        tensor_specs = [
            tf.TensorSpec(shape=(None, 1), dtype=get_dtype(feature_conf[feature]['data_type']), name=feature) for
            feature in self.features]

        @tf.function()
        def my_predict(ins):
            inputs = dict(zip(self.features, ins))
            preds = self.model(inputs)
            outputs = {}
            for target in self.targets:
                outputs['predict_{0}'.format(self.targets[target]['index'])] = preds[target]
            return outputs

        my_signatures = my_predict.get_concrete_function(tensor_specs)
        tf.keras.models.save_model(self.model, self.current_model_path, overwrite=True, include_optimizer=True,
                                   signatures=my_signatures)

    def final_save(self, day):
        """修改模型的输入和输出"""
        print("Enter the final save function")

        # 修改输入
        input_signatures = {}
        for fea in self.model.inputs:
            code = feature_name2code_dict[fea.name]
            input_dtype = fea.dtype
            input_spec = tf.TensorSpec(shape=[None, 1], dtype=input_dtype, name=str(code))
            input_signatures[fea.name] = input_spec

        # print("input_signatures: ", input_signatures)

        # 修改输出
        @tf.function()
        def predicts(input_signatures):
            outputs = self.model(input_signatures)
            print(list(outputs.keys()))
            output_signatures = {}
            output_signatures['predict_0'] = outputs['match_succ_rate']
            return output_signatures

        signatures = predicts.get_concrete_function(input_signatures)

        final_model_path = '/group/live/liuchen04/prop/black_test/model_archive/final/' + day
        print("final_model_path: ", final_model_path)
        tf.keras.models.save_model(self.model,
                                   final_model_path,
                                   overwrite=True,
                                   include_optimizer=False,
                                   signatures=signatures)

    def summary(self):
        tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.log_path, 'model_structure.png'),
                                  show_shapes=True, expand_nested=True)
        summary_file = os.path.join(self.log_path, 'summary.txt')
        with open(summary_file, 'w') as f:
            self.model.summary(print_fn=lambda x: print(x, file=f))

    def load(self):
        self.model = tf.keras.models.load_model(self.current_model_path, compile=False)
