# coding:utf-8

import tensorflow as tf

model = ""
model_savepath = ""

# 加载模型参数
model.load_weights(model_savepath + "/variables/variables").expect_partial()


feature_config = {}
model_config = {}


def build_new_input():
    """将模型的输入换成code"""
    input_signatures = {}
    for input in model.inputs:
        length = feature_config[input.name]['length']
        code = feature_config[input.name]['code']
        input_dtype = input.dtype if input.dtype != tf.float64 else tf.float32
        input_spec = tf.TensorSpec(shape=[None, length], dtype=input_dtype, name=str(code))
        input_signatures[input.name] = input_spec

    @tf.function()
    def predicts(input_signatures):
        outputs = model(input_signatures)
        output_signatures = {}
        for i, (label, conf) in enumerate(model_config['labels'].items()):
            output_signatures[f"predict_{i+1}"] = outputs[label]
            output_signatures['predict_0'] = outputs[default_output_key]
            return output_signatures

    signatures = predicts.get_concrete_function(input_signatures)
    tf.keras.models.save_model(model, filepath,
                               overwrite=True, include_optimizer=False,
                               save_traces=False, options=options,
                               signatures=signatures)

