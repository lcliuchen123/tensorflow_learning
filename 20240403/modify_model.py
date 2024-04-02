# coding:utf-8


# 修改模型的输入和输出，重新训练模型
import tensorflow as tf
path = './20240229155748'
model = tf.keras.models.load_model(path, compile=False)
feature_name2code_dict = {'fea_name': 'fea_code'}

input_signatures = {}
for fea in model.inputs:
    code = feature_name2code_dict[fea.name]
    input_dtype = fea.dtype
    input_spec = tf.TensorSpec(shape=[None, 1], dtype=input_dtype, name=str(code))
    input_signatures[fea.name] = input_spec
# print("input_signatures: ", input_signatures)


@tf.function()
def predicts(input_signatures):
    outputs = model(input_signatures)
    print(list(outputs.keys()))
    output_signatures = {}
    output_signatures['predict_0'] = outputs['match_succ_rate']
    return output_signatures


signatures = predicts.get_concrete_function(input_signatures)

#  保存到线上指定的目录
model_path = './final/2024-02-28'
tf.keras.models.save_model(model,
                           model_path,
                           overwrite=True,
                           include_optimizer=False,
                           signatures=signatures)

