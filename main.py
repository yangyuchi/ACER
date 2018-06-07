#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/23 下午1:19
# @Author  : Yang Yuchi

import threading
import numpy as np
import multiprocessing
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from network import ActorCriticNet, Worker, ep_rewards


# Hyper parameters
GAME = 'CartPole-v0'
GLOBAL_NET_SCOPE = 'global_net'
N_WORKERS = multiprocessing.cpu_count()
RANDOM_SEED = 1
BATCH_SIZE = 4
RENDER = True

tf.set_random_seed(RANDOM_SEED)

env = gym.make(GAME).unwrapped
env.seed(RANDOM_SEED)

sess = tf.Session()

with tf.device('/cpu:0'):
    global_net = ActorCriticNet(GLOBAL_NET_SCOPE, GAME)
    workers = []
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i
        workers.append(Worker(i_name, GAME, global_net))

coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())


def job():
    worker.acer_main(sess, coord)


worker_threads = []
for worker in workers:
    t = threading.Thread(target=job)
    t.start()
    worker_threads.append(t)
coord.join(worker_threads)

# for i in range(20):
#     s = env.reset()
#     while True:
#         env.render()
#         a = global_net.choose_action(s)
#         s1, r, is_done, info = env.step(a)
#         if is_done:
#             break
#         s = s1


plt.plot(np.arange(len(ep_rewards)), ep_rewards)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()













