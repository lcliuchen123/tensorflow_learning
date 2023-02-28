# coding:utf-8

# cem基础版本
# https://blog.csdn.net/mmc2015/article/details/81783448?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-81783448-blog-80567682.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-81783448-blog-80567682.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=5
# modified from https://gist.github.com/sorenbouma/6502fbf55ecdf988aa247ef7f60a9546
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.close()
# vector of means(mu) and standard dev(sigma) for each parameter
mu = np.random.uniform(size=env.observation_space.shape)
sigma = np.random.uniform(low=0.001, size=env.observation_space.shape)
print("mu: ", mu)
print("sigma: ", sigma)


def noisy_evaluation(env, W, render=False):
    """ uses parameter vector W to choose policy for 1 episode,
     returns reward from that episode"""
    reward_sum = 0
    state, _ = env.reset()
    # print("first state: ", state)
    t = 0
    while True:
        t += 1
        # print("state: ", state)
        # print("W: ", W)
        # print(np.dot(W, state))
        action = int(np.dot(W, state) > 0)  # use parameters/state to choose action
        # print(env.step(action))
        state, reward, done, info, _ = env.step(action)
        # print("after state: ", state)
        reward_sum += reward
        if render and t % 3 == 0:
            env.render()
        if done or t > 205:
            # print("finished episode, got reward:{}".format(reward_sum))
            break
    return reward_sum


def init_params(mu, sigma, n):
    """take vector of mus, vector of sigmas, create matrix such that """
    l = mu.shape[0]
    w_matrix = np.zeros((n, l))
    for p in range(l):
        w_matrix[:, p] = np.random.normal(loc=mu[p], scale=sigma[p] + 1e-17, size=(n,))
    return w_matrix


def get_constant_noise(step):
    return np.max(5 - step / 10., 0)


running_reward = 0
n = 40
p = 8
n_iter = 20
render = False

state = env.reset()
i = 0
while i < n_iter:
    # initialize an array of parameter vectors
    wvector_array = init_params(mu, sigma, n)
    reward_sums = np.zeros((n))
    for k in range(n):
        # sample rewards based on policy parameters in row k of wvector_array
        reward_sums[k] = noisy_evaluation(env, wvector_array[k, :], render)
    env.close()
    # sort params/vectors based on total reward of an episode using that policy
    rankings = np.argsort(reward_sums)

    # pick p vectors with highest reward
    top_vectors = wvector_array[rankings, :]
    top_vectors = top_vectors[-p:, :]
    print("top vectors shpae:{}".format(top_vectors.shape))

    # fit new gaussian from which to sample policy
    for q in range(top_vectors.shape[1]):
        mu[q] = top_vectors[:, q].mean()
        sigma[q] = top_vectors[:, q].std() + get_constant_noise(i)

    running_reward = 0.99 * running_reward + 0.01 * reward_sums.mean()
    print("#############################################################################")
    print("iteration:{},mean reward:{}, running reward mean:{} \n"
          " reward range:{} to {},".format(i, reward_sums.mean(),
                                           running_reward, reward_sums.min(), reward_sums.max()))
    i += 1
