# coding:utf-8

# 基于注意力的神经机器翻译的有效方法, 西班牙语到英语的seq2seq模型


import tensorflow as tf
print(tf.__version__)

import tensorflow_text as tf_text
import numpy as np
import typing
from typing import Any, Tuple
# einops是一个用于操作张量的库,比如reshape等操作
import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pathlib


class ShapeChecker():
    """检查tensor的形状 利用相同的key指定对应的维度, 比如b 表示batch, 所有有这个维度的张量的b应该一样"""
    def __init__(self):
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        # 检查当前线程是否启用了急迫执行
        if not tf.executing_eagerly():
            return
        # names 表示有几个维度, 比如
        # b = tf.constant([[1,2], [3,4]])  # <tf.Tensor: shape=(2, 2), dtype=int32
        # parsed = einops.parse_shape(tensor, 'a b')
        # parsed: {'a': 2, 'b': 2}
        parsed = einops.parse_shape(tensor, names)
        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)
            if broadcast and new_dim == 1:
                continue
            if old_dim is None:
                self.shapes[name] = new_dim
                continue
            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"found: {new_dim}"
                                 f"expected: {old_dim}\n")

# 加载指定的数据集
path_to_zip = tf.keras.utils.get_file('spa-eng.zip',
                                      origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
                                      extract=True)
path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'


def load_data(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]
    context = np.array([context for target, context in pairs])
    target = np.array([target for target, context in pairs])
    return target, context


target_raw, context_raw = load_data(path_to_file)
print(context_raw[-1])
print(target_raw[-1])

# 创建tf数据集
BUFFER_SIZE = len(context_raw)
BATCH_SIZE = 64
is_train = np.random.uniform(size=(len(target_raw),)) < 0.8
train_raw = (tf.data.Dataset.
             from_tensor_slices((context_raw[is_train], target_raw[is_train])).
             shuffle(BATCH_SIZE).
             batch(BATCH_SIZE))
val_raw = (tf.data.Dataset.
           from_tensor_slices((context_raw[~is_train], target_raw[~is_train])).
           shuffle(BUFFER_SIZE).
           batch(BATCH_SIZE))

for example_context_strings, example_target_strings in train_raw.take(1):
    print(example_context_strings[:5])
    print()
    print(example_target_strings[:5])
    break

# 文本预处理
# step1 Unicode规范化
example_text = tf.constant('¿Todavía está en casa?')
print(example_text.numpy())
print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())


def tf_lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # 添加空格
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join('[START]', text, '[END]', separator=' ')
    return text


print(example_text.numpy().decode())
print(tf_lower_and_split_punct(example_text).numpy().decode())


# 文本矢量化
max_vocab_size = 5000
context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True
)
context_text_processor.adapt(train_raw.map(lambda context, target: context))
# 前10个词
print(context_text_processor.get_vocabulary()[:10])

target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True
)
target_text_processor.adapt(train_raw.map(lambda context, target: target))
print(target_text_processor.get_vocabulary()[:10])

# 字符转化为id
example_tokens = context_text_processor(example_context_strings)
print(example_tokens[:3, :])
context_vocab = np.array(context_text_processor.get_vocabulary())
# id转回文本
tokens = context_vocab[example_tokens[0].numpy()]
print(' '.join(tokens))

plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens.to_tensor())
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens.to_tensor() != 0)
plt.title('Mask')
plt.show()


# 用0进行mask
def process_text(context, target):
    # 去除特殊字符，统一编码，映射成id
    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)
    targ_in = target[:, :-1]
    targ_out = target[:, 1:]
    return (context, targ_in), targ_out


train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)
for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
    print(ex_context_tok[0, :10].numpy())
    print()
    print(ex_tar_in[0, :10].numpy())
    print(ex_tar_out[0, :10].numpy())


# 编码器
UNITS = 256


class Encoder(tf.keras.layer.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units
        # embedding层转化tokens到ids
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        # 双向GRU
        self.rnn = tf.keras.layers.Bidirectional(
            merger_mode='sum',
            layer=tf.keras.layers.GRU(units, return_sequences=True, recurrent_initializer='glorot_uniform'))

    def call(self, x):
        shape_checker = ShapeChecker(x, 'batch s')

        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        x = self.rnn(x)
        shape_checker(x, 'batch s units')
        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            # tf.newaxis添加维度
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context


# 测试
encoder = Encoder(context_text_processor, UNITS)
ex_context = encoder(ex_context_tok)


# 注意力层
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True
        )
        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        # 缓存权重
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


attention_layer = CrossAttention(UNITS)
# 编码器的注意力层测试
embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(),
                                  output_dim=UNITS,
                                  mask_zero=True)
ex_tar_embed = embed(ex_tar_in)
result = attention_layer(ex_tar_embed, ex_context)
print(f'Context sequence, shape (batch, s, units): {ex_context.shape}')
print(f'Target sequence, shape (batch, t, units): {ex_tar_embed.shape}')
print(f'Attention result, shape (batch, t, units): {result.shape}')
print(f'Attention weights, shape (batch, t, s):    {attention_layer.last_attention_weights.shape}')

# 上下文权重求和，均为1
print(attention_layer.last_attention_weights[0].numpy().sum(axis=-1))

attention_weights = attention_layer.last_attention_weights
mask = (ex_context_tok != 0).numpy()

plt.subplot(1, 2, 1)
plt.pcolormesh(mask * attention_weights[:, 0, :])
plt.title('Attention weights')

plt.subplot(1, 2, 2)
plt.pcolormesh(mask)
plt.title('Mask')


# 解码器(单向RNN)
class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        """添加函数方法"""
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]'
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True
        )
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')
        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zeros=True)
        self.rnn = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attention = CrossAttention(units)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)


@Decoder.add_method
def call(self, context, x, state=None, return_state=False):
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t')
    shape_checker(context, 'batch s units')

    # x是目标序列的输入
    x = self.embedding(x)
    shape_checker(x, 'batch t units')

    # 预处理目标序列
    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'batch t units')

    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'batch t s')
    shape_checker(self.last_attention_weights, 'batch t s')

    logits = self.output_layer(x)
    shape_checker(logits, 'batch t target_vocab_size')

    # state: 解码器之前的输出（解码器 RNN 的内部状态）。传递上一次运行的状态以继续在上次停止的地方生成文本。
    if return_state:
        return logits, state
    else:
        return logits


# 测试
decoder = Decoder(target_text_processor, UNITS)
logits = decoder(ex_context, ex_tar_in)
print(f'encoder output shape: (batch, s, units) {ex_context.shape}')
print(f'input target tokens shape: (batch, t) {ex_tar_in.shape}')
print(f'logits shape shape: (batch, target_vocabulary_size) {logits.shape}')


# 推理
@Decoder.add_method
def get_initial_state(self, context):
    """获取start对应的张量"""
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$','')
    return result


@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature=0.0):
    logits, state = self(
        context, next_token,
        state= state,
        return_state=False
    )

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :] / temperature
        next_token = tf.random.categorical(logits, num_samples=1)
        # 如果一个序列输出的是end_token, 设置为done
        done = done | (next_token == self.end_token)
        # 如果序列输出结束，就用0填充
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
        return next_token, done, state


next_token, done, state = decoder.get_initial_state(ex_context)
tokens = []
for n in range(10):
    next_token, done, state = decoder.get_next_token(
        ex_context, next_token, done, state, temperature=1.0)
    tokens.append(next_token)

tokens = tf.concat(tokens, axis=-1)
result = decoder.tokens_to_text(tokens)
print(result[:3].numpy())


# 构造模型并训练
class Translator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units, context_text_processor, target_text_processor):
        super().__init__()
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits


model = Translator(UNITS, context_text_processor, target_text_processor)
logits = model(ex_context_tok, ex_tar_in)
print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')


def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_acc, masked_loss])
vocab_size = 1.0 * target_text_processor.vocabulary_size()
print({"expected_loss": tf.math.log(vocab_size).numpy(),
       "expected_acc": 1/vocab_size})
model.evaluate(val_ds, steps=20, return_dict=True)
history = model.fit(train_ds.repeat(),
                    epochs=100,
                    steps_per_epoch=100,
                    validation_data=val_ds,
                    validation_steps=20,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

# 绘制loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/TOKEN')
plt.legend()
plt.show()

plt.plot(history.history['masked_acc'], label='accuracy')
plt.plot(history.history['val_masked_acc'], label='val_accuracy')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/TOKEN')
plt.legend()
plt.show()


# 模型训练完毕后，编写函数实现完整的翻译
@Translator.add_method
def translate(self, texts, *, max_length=50, temperature=0.0):
    context = self.encoder.convert_input(texts)
    batch_size = tf.shape(context)[0]
    tokens = []
    attention_weights = []
    next_token, done, state = self.decoder.get_initial_state(context)
    for _ in range(max_length):
        next_token, done, state = self.decoder.get_next_token(
            context, next_token, done, state, temperature
        )
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)
        if tf.executing_eagerly() and tf.reduce_all(done):
            break
    tokens = tf.concat(tokens, axis=-1)  # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)
    result = self.decoder.tokens_to_text(tokens)
    return result


result = model.translate(['¿Todavía está en casa?'])
print(result[0].numpy().decode())


# 生成注意力图
@Translator.add_method
def plot_attention(self, text, **kwargs):
    assert isinstance(text, str)
    output = self.translate([text], **kwargs)
    output = output[0].numpy().decode()
    attention = self.last_attention_weights[0]
    context = tf_lower_and_split_punct(text)
    context = context.numpy().decode().split()
    output = tf_lower_and_split_punct(output)
    output = output.numpy().decode().split()[1:]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis', vmin=0.0)
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + output, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel('Input text')
    ax.set_ylabel('Output label')


model.plot_attention('¿Todavía está en casa?')

# This is my life.
model.plot_attention('Esta es mi vida.')

# Try to find out.'
model.plot_attention('Tratar de descubrir.')

long_text = context_raw[-1]
import textwrap
print("Excepted output:\n", '\n'.join(textwrap.wrap(target_raw[-1])))
model.plot_attention(long_text)

# 一次性多预测几个
inputs = [
    'Hace mucho frio aqui.', # "It's really cold here."
    'Esta es mi vida.', # "This is my life."
    'Su cuarto es un desastre.' # "His room is a mess"
]
for t in inputs:
    print(model.translate([t])[0].numpy().decode())
print()

result = model.translate(inputs)
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()


# 导出模型
class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None,])])
    def translate(self, inputs):
        return self.model.translate(inputs)


export = Export(model)
_ = export.translate(tf.constant(inputs))
result = export.translate(tf.constant(inputs))
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()

# 保存模型
tf.saved_model.save(export, 'translator', signatures={'serving_default': export.translate})

reloaded = tf.saved_model.load('translator')
_ = reloaded.translate(tf.constant(inputs))   # warmup

result = reloaded.translate(tf.constant(inputs))
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()


# 利用动态循环进行预测
@Translator.add_method
def translate(self, texts, *, max_length=500, temperature=tf.constant(0.0)):
    shape_checker = ShapeChecker()
    context = self.encoder.convert_input(texts)
    batch_size = tf.shape(context)[0]
    shape_checker(context, 'batch s units')

    next_token, done, state = self.decoder.get_initial_state(context)
    # 动态改变长度
    tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)
    for t in range(max_length):
        next_token, done, state = self.decoder.get_next_token(context, next_token, done, state, temperature)
        shape_checker(next_token, 'batch t1')
        tokens = tokens.write(t, next_token)

        if tf.reduce_all(done):
            break
    tokens = tokens.stack()
    shape_checker(tokens, 't batch t1')
    tokens = einops.rearrange(tokens, 't batch 1 -> batch t')
    shape_checker(tokens, 'batch t')
    text = self.decoder.tokens_to_text(tokens)
    shape_checker(text, 'batch')
    return text


result = model.translate(inputs)
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()

class Export(tf.Module):
    def __init__(self, model):
        self.model =model
    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None,])])
    def translate(self, inputs):
        return self.model.translate(inputs)

export = Export(model)
_ = export.translate(inputs)
result = export.translate(inputs)
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()


tf.saved_model.save(export, 'dynamic_translator', signatures={'serving_default':export.translate})
reloaded = tf.saved_model.load('dynamic_translator')
_ = reloaded.translate(tf.constant(inputs))  # warmup
result = reloaded.translate(tf.constant(inputs))
print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()



