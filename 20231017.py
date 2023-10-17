# coding:utf-8

# 强化学习 ac结构

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

env = gym.make("CartPole-v0")
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# 特别小的一个值
eps = np.finfo(np.float32).eps.item()


# 定义模型
class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions, num_hidden_units):
        super().__init__()
        self.common = layers.Dense(num_hidden_units, activation='relu')
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


num_actions = env.action_space.n
print("num_actions: ", num_actions)
num_hidden_units = 128
model = ActorCritic(num_actions, num_hidden_units)


# 1、收集训练数据
def env_step(action):
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action):
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


# tf.TensorArray() 可变长度数组
def run_episode(initial_state, model, max_steps):
    """run a single episode to collect training data"""
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    initial_state_shape = initial_state.shape
    state = initial_state
    for t in tf.range(max_steps):
        # covert to a batch tensor [batch_size =1]
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # 随机采样下一个动作
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # 保存
        values = values.write(t, tf.squeeze(value))
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # 获取下一个state和reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break
    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    return action_probs, values, rewards


# 2、计算预期回报
def get_expected_return(rewards, gamma, standardize=True):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]
    # 标准化
    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
    return returns


# 3、actor-critic损失
# critic损失是回归损失函数 huber_loss
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(action_probs, values, returns):
    advantage = returns - values
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    critic_loss = huber_loss(values, returns)
    return actor_loss + critic_loss


# 4、定义训练步骤更新参数
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
@tf.function
def train_step(initial_state, model, optimizer, gamma, max_steps_per_episode):
    # 自定义梯度
    with tf.GradientTape() as tape:
        # 获取训练数据
        action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)

        # 计算预估的rewards
        returns = get_expected_return(rewards, gamma)

        # 转换训练数据格式为tf需要的格式
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # 计算loss更新网络
        loss = compute_loss(action_probs, values, returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward


# 5、运行训练循环
min_episode_criterion = 100
max_episodes = 10000
max_steps_per_episodes = 1000
reward_threshold = 195
running_reward = 0
gamma = 0.99

episodes_reward: collections.deque = collections.deque(maxlen=min_episode_criterion)
with tqdm.trange(max_episodes) as t:
    for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episodes))
        episodes_reward.append(episode_reward)
        # statistics.mean 给定数据集的平均值
        running_reward = statistics.mean(episodes_reward)
        t.set_description(f'Episode {i}')
        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

        if i % 10 == 0:
            pass
        if running_reward > reward_threshold and i >= min_episode_criterion:
            break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

# 可视化  版本不匹配无法正确显示
from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display

display = Display(visible=0, size=(400, 300))
display.start()


def render_episode(env, model, max_steps):
    screen = env.render(mode='rgb_array')
    im = Image.fromarray(screen)
    images = [im]
    state = tf.constant(env.reset()[0], dtype=tf.float32)
    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        action_probs, _ = model(state)
        action = np.argmax(np.squeeze(action_probs))
        state, _, done, _, _ = env.step(action)
        state = tf.constant(state, dtype=tf.float32)

        if i % 10 == 0:
            screen = env.render(mode='rgb_array')
            images.append(Image.fromarray(screen))
        if done:
            break
    return images


images = render_episode(env, model, max_steps_per_episodes)
image_file = 'cartpole-v1.gif'
images[0].save(image_file, save_all=True, append_images=images[1:], loop=0, duration=1)

import tensorflow_docs.vis.embed as embed
embed.embed_file(image_file)






