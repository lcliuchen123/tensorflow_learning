# coding:utf-8
# 测试负采样，参考https://zhuanlan.zhihu.com/p/489022692
# 重要性采样参考https://blog.csdn.net/wangpeng138375/article/details/75151064

import tensorflow as tf
print(tf.__version__)
import numpy as np


def gen_neg(num_sampled):
    seed = 1234
    context_class = tf.reshape(tf.constant(0, dtype='int64'), (1, 1))
    neg_sampled_candidates, true_expected_count, sampled_expected_count = \
        tf.random.log_uniform_candidate_sampler(true_classes=context_class,
                                                num_true=1,
                                                num_sampled=num_sampled,
                                                unique=True,
                                                range_max=8,
                                                seed=seed)
    print("neg_sampled_candidates: ", neg_sampled_candidates.numpy())
    print("true_expected_count: ", true_expected_count.numpy())
    print("sampled_expected_count: ", sampled_expected_count.numpy())


if __name__ == "__main__":
    pro_list = []
    for i in range(8):
        pro_one = (np.log(i + 2) - np.log(i + 1)) / np.log(8 + 1)
        pro_list.append(pro_one)

    print("pro_list: ", pro_list)

    # print("----------num=1---------------")
    # for i in range(10):
    #     gen_neg(1)

    print("----------num=2---------------")
    for i in range(10):
        gen_neg(2)
