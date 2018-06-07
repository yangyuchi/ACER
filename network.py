#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 下午7:47
# @Author  : Yang Yuchi


import tensorflow as tf
import numpy as np
import gym
import math
import random
from replay_buffer import ReplayMemory


N_HIDDEN = 200
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.001
ENTROPY_BETA = 0.001
GAMMA = 0.9
global_iter = 0
ep_rewards = []
MAX_ITERATION = 10000
UPDATE_FREQ = 10
MAX_STORED_EPISODES = 4000
MAX_EPISODE_LENGTH = 1000
REPLAY_RATIO = 4


class ActorCriticNet(object):
    def __init__(self, scope, game):
        self.env = gym.make(game).unwrapped
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.scope = scope
        self.s = tf.placeholder(tf.float32, [None, self.input_size], 'states')
        with tf.variable_scope(self.scope):
            self.a_prob, self.v, self.a_params, self.c_params = self._build_net()
            self.target_v = tf.placeholder(tf.float32, [None, 1], 'target_value')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'history_actions')
            self.rho = tf.placeholder(tf.float32, [None, 1], 'importance_sampling')
            td_error = tf.subtract(self.target_v, self.v, name='td_error')
            self.c_loss = tf.reduce_mean(tf.square(td_error))
            neg_log_prob = tf.reduce_sum(-tf.log(self.a_prob) *
                                         tf.one_hot(self.a_his, self.output_size, dtype=tf.float32),
                                         axis=1,
                                         keep_dims=True)
            self.a_loss = tf.reduce_mean(neg_log_prob * td_error)

    def _build_net(self):
        w_initializer = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope('actor'):
            l1 = tf.layers.dense(self.s, N_HIDDEN, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='l1')
            a_prob = tf.layers.dense(l1, self.output_size, tf.nn.softmax,
                                     kernel_initializer=w_initializer, name='l2_actor')
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.s, N_HIDDEN, tf.nn.relu,
                                 kernel_initializer=w_initializer, name='l1')
            v = tf.layers.dense(l1, 1, kernel_initializer=w_initializer, name='l2_critic')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')
        return a_prob, v, a_params, c_params

    def choose_action(self, s, sess):
        s = s[np.newaxis, :]
        a_prob = sess.run(self.a_prob, {self.s: s})
        return np.random.choice(np.arange(a_prob.shape[1]), p=a_prob.ravel()), np.arange(a_prob.shape[1])


class Worker(ActorCriticNet):
    def __init__(self, scope, game, global_net: ActorCriticNet):
        self.ep_num = 0
        self.name = scope
        self.memory = ReplayMemory(MAX_STORED_EPISODES, MAX_EPISODE_LENGTH)
        super(Worker, self).__init__(scope, game)
        with tf.name_scope('local_gradients'):
            self.a_gradients = tf.gradients(self.a_loss, self.a_params)
            self.c_gradients = tf.gradients(self.c_loss, self.c_params)
        with tf.name_scope('pull'):
            self.pull_a_params_op = [tf.assign(l_p, g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
            self.pull_c_params_op = [tf.assign(l_p, g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
        with tf.name_scope('update'):
            self.train_op_a = tf.train.RMSPropOptimizer(LEARNING_RATE_ACTOR, name='RMSProp_Actor')
            self.train_op_c = tf.train.RMSPropOptimizer(LEARNING_RATE_CRITIC, name='RMSProp_Critic')
            self.update_a_op = self.train_op_a.apply_gradients(zip(self.a_gradients, global_net.a_params))
            self.update_c_op = self.train_op_c.apply_gradients(zip(self.c_gradients, global_net.c_params))

    def pull_global(self, sess):
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def update_global(self, sess, feed_dict):
        sess.run([self.update_a_op, self.update_c_op], feed_dict)

    def acer_main(self, sess, coord):
        global global_iter
        while not coord.should_stop() and global_iter < MAX_ITERATION:
            self._train(sess, True)
            n = self._get_poisson(REPLAY_RATIO)
            for i in range(n):
                self._train(sess, False)

    def _train(self, sess, on_policy: bool):
        # call on policy part, generate episode and save to replay memory
        if on_policy:
            ep_r = 0
            s = self.env.reset()
            while True:
                a, p = self.choose_action(s, sess)
                next_s, r, is_done, info = self.env.step(a)
                if is_done:
                    r = -5
                ep_r += r
                self.memory.push(s, a, r, p, is_done)
                if is_done:
                    break
                s = next_s
            self.ep_num += 1
            print('Thread:', self.name, 'Episode number:', self.ep_num, 'Reward:', ep_r)
        # call off policy part
        else:
            samples = self.memory.sample(UPDATE_FREQ)
            buffer_s = [m[0] for m in samples]
            buffer_a = [m[1] for m in samples]
            buffer_r = [m[2] for m in samples]
            buffer_p = [m[3] for m in samples]
            buffer_done = [m[4] for m in samples]

            buffer_rho = []
            for i in range(len(samples)):
                pi = sess.run(self.a_prob, {self.s: buffer_s[i][np.newaxis, :]})[0, buffer_a[i]]
                mu = buffer_p[i][buffer_a[i]]
                buffer_rho.append(pi/mu)

            if buffer_done[-1]:
                q_ret = 0
            else:
                q_ret = sess.run(self.v, {self.s: buffer_s[-1][np.newaxis, :]})[0, 0]
            buffer_q_ret = []
            for r_i in reversed(buffer_r):
                q_ret = r_i + GAMMA * q_ret
                buffer_q_ret.append(q_ret)
            buffer_q_ret.reverse()
            buffer_s, buffer_a, buffer_q_ret, buffer_rho = \
                np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_q_ret), np.vstack(buffer_rho)
            feed_dict = {
                self.s: buffer_s,
                self.a_his: buffer_a,
                self.target_v: buffer_q_ret,
                self.rho: buffer_rho
            }
            self.update_global(sess, feed_dict)
            self.pull_global(sess)

    # poisson random number by Knuth
    def _get_poisson(self, r):
        l, k, p = math.exp(-r), 0, 1
        while True:
            k += 1
            p *= random.uniform(0, 1)
            if not p > l:
                break
        return k - 1



