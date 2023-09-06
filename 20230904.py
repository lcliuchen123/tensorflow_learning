# coding:utf-8

# word2vec的skip-gram

import io
import re
import string
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print(len(tokens))

# 创建索引
vocab, index = {}, 1
vocab['<pad>'] = 0
for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index += 1
vocab_size = len(vocab)

inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)

# 向量化
example_sequences = [vocab[word] for word in tokens]
print(example_sequences)

# word2vec数据准备
# 获取正样本 tf.keras.preprocessing.sequence.skipgrams
window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
    example_sequences,
    vocabulary_size=vocab_size,
    window_size=window_size,
    negative_samples=0
)
print(len(positive_skip_grams))

for target, context in positive_skip_grams[:5]:
    print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

# 负采样
# 正样本
target_word, context_word = positive_skip_grams[0]

# 负采样 tf.random.log_uniform_candidate_sampler
num_ns = 4
context_class = tf.reshape(tf.constant(context_word, dtype='int64'), (1, 1))
negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class,  # 正例
    num_true=1,  # 正样本数量
    num_sampled=num_ns,  # 负样本数量
    unique=True,  # 负样本没有重复的
    range_max=vocab_size,  # 从[0, vocab_size]里面采样
    seed=SEED,
    name='negative_sampling'
)
print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

# 构建一个训练示例

# 减少一个维度，方便后续步骤的concat
squeezed_context_class = tf.squeeze(context_class, 1)

# 合并正负样本
context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)

label = tf.constant([1] + [0] * num_ns, dtype="int64")
target = target_word

print(f"target_index: {target}")
print(f"target_word: {inverse_vocab[target_word]}")
print(f"context_indices: {context}")
print(f"context_words: {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label: {label}")
print("target: ", target)
print("context: ", context)
print("label: ", label)

# tf.keras.preprocessing.sequence.make_sampling_table生成基于词频排名的概率采样表
sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
print(sampling_table)


# 生成训练数据
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=vocab_size)

    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0
        )

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype='int64'), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name='negative_sampling'
            )
            context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
            label = tf.concat([1] + [0] * num_ns, dtype='int64')

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    return targets, contexts, labels


# 从数据集获取序列
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
with open(path_to_file, 'r') as f:
    lines = f.read().splitlines()
    for line in lines[:20]:
        print(line)
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x)), bool)


def custom_standardization(input_data):
    """清除标点符号
       string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    """
    lowercase = tf.string.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')


# 文本向量化层: layers.TextVectorization
vocab_size = 4096
sequence_length = 10
vectorize_layer = layers.TextVectorization(
    standaridize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)
vectorize_layer.adapt(text_ds.batch(1024))
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))
for seq in sequences[:5]:
    print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

# 从序列生成训练示例
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED
)
targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)
print("\n")
print(f"target.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

# 配置数据集提高性能
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)
dataset = dataset.cache().prefectch(buffsize=AUTOTUNE)
print(dataset)


# 定义模型
class word2vec(tf.keras.model):
    def __init__(self, vocab_size, embedding_dim):
        super(word2vec, self).__int__()
        self.target_embedding = layers.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name='w2v_embedding'
        )
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=num_ns+1)

    def call(self, pair):
        target, context = pair
        # target: (batch,)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # word_emb: (batch, embed)
        word_emb = self.target_embedding(target)
        # context_emb: (batch, context, embed)
        context_emb = self.context_embedding(context)
        # dots: (batch, context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots


# 定义损失函数并编译模型
def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


embedding_dim = 128
word2vec = word2vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

# 嵌入查找和分析
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
for index, word in enumerate(vocab):
    if index == 0:
        continue
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')

out_m.close()
out_v.close()




