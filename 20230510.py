# coding:utf-8

import tensorflow as tf
import tensorflow_datasets as tfds
import os

print(tf.__version__)

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

# 分布式训练
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy']
                  )


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif 3 <= epoch < 7:
        return 1e-4
    else:
        return 1e-5


# 自定义回调函数
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """epoch结束的时候"""
        print('\nLearning rate for epoch {} is {}'.format(epoch+1, model.optimizer.lr.numpy()))


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

# 训练和评估
EPOCHS = 12
model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)

# 加载模型
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
eval_loss, eval_acc = model.evaluate(eval_dataset)
print("Eval loss: {}, Eval accuracy:{} ".format(eval_loss, eval_acc))

# 导出到Savedmodel
path = 'saved_model'
model.save(path, save_format='tf')

# 在没有Strategy.scope的情况下加载模型
unreplicated_model = tf.keras.models.load_model(path)
unreplicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])
eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)
print('Eval_loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


# 使用Strategy.scope加载模型
with strategy.scope():
    replicated_model = tf.keras.models.load_model(path)
    replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=['accuracy'])
    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


