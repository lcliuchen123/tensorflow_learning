# coding:utf-8

# 自定义负采样
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
import tensorflow_recommenders_addons as tfra


def my_sampled_softmax_loss(weights,
                         biases,
                         labels,
                         inputs,
                         num_sampled,
                         num_classes,
                         num_true=1,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         name="sampled_softmax_loss",
                         use_tfra=False,
                         seed=None):
    logits, labels = _compute_sampled_logits(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        subtract_log_q=True,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name,
        seed=seed,
        use_tfra=use_tfra)
    labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")
    sampled_losses = nn_ops.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    return sampled_losses


def _sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    cols = array_ops.shape(x)[1]
    ones_shape = array_ops.stack([cols, 1])
    ones = array_ops.ones(ones_shape, x.dtype)
    return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            seed=None,
                            use_tfra=False):
    """
            输入：
             weights: 待优化的矩阵，形状[num_classes, dim]。可以理解为所有item embedding矩阵，那时num_classes=所有item的个数
             biases: 待优化变量，[num_classes]。每个item还有自己的bias，与user无关，代表自己的受欢迎程度。
             labels: 正例的item ids，形状是[batch_size,num_true]的正数矩阵。每个元素代表一个用户点击过的一个item id。允许一个用户可以点击过多个item。
             inputs: 输入的[batch_size, dim]矩阵，可以认为是user embedding
             num_sampled：整个batch要采集多少负样本
             num_classes: 在u2i中，可以理解成所有item的个数
             num_true: 一条样本中有几个正例，一般就是1
             subtract_log_q：是否要对匹配度，进行修正
             remove_accidental_hits：如果采样到的某个负例，恰好等于正例，是否要补救
            输出：
             out_logits: [batch_size, num_true + num_sampled]
             out_labels: 与`out_logits`同形状
    """
    # 判断数据类型是否符合预期
    if isinstance(weights, variables.PartitionedVariable):
        weights = list(weights)
    if not isinstance(weights, list):
        weights = [weights]

    with ops.name_scope(name, "compute_sampled_logits", weights + [biases, inputs, labels]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   labels: [batch_size, num_true]
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
        sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    # sampled: 负样本的id, true_expected_count:正例在log-uniform采样分布中的概率，接下来修正logit时用得上,
    # sampled_expected_count: 负例在log-uniform采样分布中的概率，接下来修正logit时用得上
    sampled, true_expected_count, sampled_expected_count = (
        array_ops.stop_gradient(s) for s in sampled_values)
    # pylint: enable=unpacking-non-sequence
    sampled = math_ops.cast(sampled, dtypes.int64)

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    # 融合正负例的ids, 形状为batch_size * num_true + num_sampled
    all_ids = array_ops.concat([labels_flat, sampled], 0)

    # Retrieve the true weights and the logits of the sampled weights.
    # weights shape is [num_classes, dim]
    # 改动1:
    if use_tfra:
        all_w, _ = tfra.dynamic_embedding.embedding_lookup_unique(params=weights,
                                                                  ids=all_ids,
                                                                  name="liuchen04_softmax_w_t",
                                                                  return_trainable=True,
                                                                  partition_strategy=partition_strategy)
    else:
        all_w = embedding_ops.embedding_lookup(weights, all_ids, partition_strategy=partition_strategy)

    if all_w.dtype != inputs.dtype:
        all_w = math_ops.cast(all_w, inputs.dtype)

    # true_w shape is [batch_size * num_true, dim]
    # 从all_w中抽取出对应正例的item embedding
    true_w = array_ops.slice(all_w, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))
    # 从all_w中抽取出对应负例的item embedding
    sampled_w = array_ops.slice(all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # Apply X*W', which yields [batch_size, num_sampled]
    # 计算与负例的匹配度
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    # Retrieve the true and sampled biases, compute the true logits, and
    # add the biases to the true and sampled logits.
    # 改动2：
    if use_tfra:
        all_b, _ = tfra.dynamic_embedding.embedding_lookup_unique(params=biases,
                                                                  ids=all_ids,
                                                                  name="liuchen04_softmax_w_b",
                                                                  return_trainable=True,
                                                                  partition_strategy=partition_strategy)
        all_b = array_ops.reshape(all_b, [-1])
    else:
        all_b = embedding_ops.embedding_lookup(biases, all_ids, partition_strategy=partition_strategy)  # shape=(None,)
    if all_b.dtype != inputs.dtype:
        all_b = math_ops.cast(all_b, inputs.dtype)
    # true_b is a [batch_size * num_true] tensor
    # sampled_b is a [num_sampled] float tensor
    # 改动3
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    # 计算正例的匹配度
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

    true_b = array_ops.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b

    # 如果采样到的负例，恰好也是正例，就要补救
    if remove_accidental_hits:
        # 计算负采样中命中的正例的索引(indices), ids, 权重（-FLOAT_MAX）
        acc_hits = candidate_sampling_ops.compute_accidental_hits(labels, sampled, num_true=num_true)
        acc_indices, acc_ids, acc_weights = acc_hits

        # This is how SparseToDense expects the indices.
        acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
        acc_ids_2d_int32 = array_ops.reshape(math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
        sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1, "sparse_indices")
        sampled_logits_shape = array_ops.concat([array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)], 0)

        if sampled_logits.dtype != acc_weights.dtype:
            acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
            # 补救方法是在冲突的位置(sparse_indices)的负例logits(sampled_logits)
            # 加上一个非常大的负数acc_weights（值为-FLOAT_MAX）
            # 这样在计算softmax时，相应位置上的负例对应的exp值=0，就不起作用了
            sampled_logits += gen_sparse_ops.sparse_to_dense(
                sparse_indices,
                sampled_logits_shape,
                acc_weights,
                default_value=0.0,
                validate_indices=False)

    if subtract_log_q:
        # 对匹配度做修正，对应上边公式中的
        # G(x,y)=F(x,y)-log Q(y|x)
        # item热度越高，被修正得越多
        true_logits -= math_ops.log(true_expected_count)
        sampled_logits -= math_ops.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    # 如果num_true=n，那么每行样本的label就是[1/n,1/n,...,1/n,0,0,...,0]的形式
    # 对于下游的sigmoid loss或softmax loss，属于soft label
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)], 1)

    return out_logits, out_labels
