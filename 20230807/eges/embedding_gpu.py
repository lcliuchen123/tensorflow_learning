# -*- coding: utf-8 -*-

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import tensorflow_recommenders_addons as tfra


class EmbeddingLayerGPU(tfra.dynamic_embedding.keras.layers.BasicEmbedding):
    def call(self, ids):
        with tf.name_scope(self.name + "/EmbeddingLookupUnique"):
            ids = tf.convert_to_tensor(ids)
            shape = tf.shape(ids)
            ids_flat = tf.reshape(ids, tf.reduce_prod(shape, keepdims=True))
            unique_ids, idx = tf.unique(ids_flat)
            unique_embeddings = tfra.dynamic_embedding.shadow_ops.embedding_lookup(self.shadow, unique_ids)
            embeddings_flat = tf.gather(unique_embeddings, idx)
            embeddings_shape = tf.concat(
                [shape, [self.embedding_size]], 0)
            embeddings = tf.reshape(embeddings_flat, embeddings_shape)
            return embeddings

