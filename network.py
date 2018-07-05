#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 下午7:47
# @Author  : Yang Yuchi


import tensorflow as tf
import numpy as np
import gym
import math
import random
from replay_memory import ReplayMemory

global_iter = 0
ep_num = 0
ep_rewards = []


class ActorCriticNet(object):
    def __init__(self, args, scope, sess=None):
        self.sess = sess
        self.env = gym.make(args.env).unwrapped
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        self.scope = scope
        self.s = tf.placeholder(tf.float32, [None, self.input_size], 'states')
        with tf.variable_scope(self.scope):
            self.a_prob, self.v, self.q, self.a_params, self.c_params = self._build_net(args)
            self.target = tf.placeholder(tf.float32, [None, 1], 'target_value')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'history_actions')
            self.rho = tf.placeholder(tf.float32, [None, 1], 'importance_sampling')
            self.average_p = tf.placeholder(tf.float32, [None, self.output_size], 'average_policy')

    def _build_net(self, args):
        with tf.variable_scope('actor'):
            l1 = tf.layers.dense(self.s, args.num_hidden, tf.nn.relu, name='l1')
            a_prob = tf.layers.dense(l1, self.output_size, tf.nn.softmax, name='l2_actor')

        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.s, args.num_hidden, tf.nn.relu, name='l1')
            q = tf.layers.dense(l1, self.output_size, name='l2_q')
            v = tf.reduce_sum(q * a_prob)  # V is expectation of Q under π

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')
        return a_prob, v, q, a_params, c_params

    #  choose action greedily
    def choose_action(self, s, greedy=True):
        s = s[np.newaxis, :]
        a_prob = self.sess.run(self.a_prob, {self.s: s})
        if greedy:
            return np.argmax(a_prob.ravel())
        else:
            return np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())

    def get_policy(self, s):
        s = s[np.newaxis, :]
        a_prob = self.sess.run(self.a_prob, {self.s: s})
        return a_prob.ravel()


class Agent(ActorCriticNet):
    def __init__(self, args, rank, scope, global_net: ActorCriticNet, average_net: ActorCriticNet, sess=None):
        super(Agent, self).__init__(args, scope, sess)
        self.env.seed(args.seed + rank)
        tf.set_random_seed(args.seed + rank)
        self.memory = ReplayMemory(args.memory_capacity, args.max_stored_episode_length)
        self.name = scope
        with tf.variable_scope(self.scope):
            advantage = tf.subtract(self.target, self.v, name='advantage')
            self.c_loss = 0.5 * tf.reduce_mean(tf.square(advantage))  # minimize advantage
            # avoid NaN with clipping when value in policy becomes zero, or use
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l2, labels=self.a_his)
            neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.a_prob, 1e-20, 1.0)) *
                                         tf.one_hot(self.a_his, self.output_size, dtype=tf.float32),
                                         axis=1,
                                         keepdims=True)
            # maximize total reward (log_p * R) is minimize -(log_p * R), tf can only minimize the loss
            self.a_loss = self.rho * tf.reduce_mean(neg_log_prob * advantage)
            self.kl_loss = -self._kl_div(self.average_p, self.a_prob)

        with tf.name_scope('local_gradients'):
            self.a_gradients = tf.gradients(self.a_loss, self.a_params)
            kl_gradients = tf.gradients(self.kl_loss, self.a_params)
            # Compute dot products of gradients
            k_dot_g = sum(tf.reduce_sum(k_g * a_g) for k_g, a_g in zip(kl_gradients, self.a_gradients))
            k_dot_k = sum(tf.reduce_sum(k_g ** 2) for k_g in kl_gradients)
            # Compute trust region update
            zero = tf.zeros_like(k_dot_k)
            trust_factor = tf.where(tf.greater(k_dot_k, zero), tf.maximum((k_dot_g - args.delta) / k_dot_k, 0), zero)
            self.total_gradients = [g_p - trust_factor * k_p for g_p, k_p in zip(self.a_gradients, kl_gradients)]
            self.total_gradients = [tf.clip_by_norm(gradient, args.max_gradient_norm) for gradient in self.total_gradients]
            self.c_gradients = tf.gradients(self.c_loss, self.c_params)
            self.c_gradients = [tf.clip_by_norm(gradient, args.max_gradient_norm) for gradient in self.c_gradients]
        with tf.name_scope('pull'):
            self.pull_a_params_op = [tf.assign(l_p, g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
            self.pull_c_params_op = [tf.assign(l_p, g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
        with tf.name_scope('update'):
            self.optimizer_a = tf.train.RMSPropOptimizer(args.learning_rate, name='Optimizer_Actor')
            self.optimizer_c = tf.train.RMSPropOptimizer(args.learning_rate, name='Optimizer_Critic')
            self.update_a_op = self.optimizer_a.apply_gradients(zip(self.total_gradients, global_net.a_params))
            self.update_c_op = self.optimizer_c.apply_gradients(zip(self.c_gradients, global_net.c_params))
            # Update shared average policy network
            self.update_average_net_op = [tf.assign(a_p, args.alpha * a_p + (1 - args.alpha) * g_p)
                                          for a_p, g_p in zip(average_net.a_params, global_net.a_params)]
    #         self.test1 = average_net.a_params
    #         self.test2 = global_net.a_params
    #
    # def test(self):
    #     print(self.sess.run(self.test1)[3])
    #     print(self.sess.run(self.test2)[3])

    def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def update_global(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    def update_average(self):
        self.sess.run(self.update_average_net_op)

    def acer_main(self, args, coord, average_net: ActorCriticNet):
        global global_iter
        while not coord.should_stop() and global_iter < args.max_iterations:
            self._train(args, average_net, on_policy=True)
            n = self._get_poisson(args.replay_ratio)
            if len(self.memory) >= args.replay_start:
                for i in range(n):
                    self._train(args, average_net, on_policy=False)

    def _train(self, args, average_net: ActorCriticNet, on_policy: bool):
        global ep_num
        t = 1
        # call on policy part, generate episode and save to replay memory
        if on_policy:
            total_reward = 0
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r, buffer_average_p, buffer_rho = [], [], [], [], []
            while True:
                action = self.choose_action(state, greedy=False)
                policy = self.get_policy(state)
                next_state, reward, is_done, _ = self.env.step(action)
                average_policy = average_net.get_policy(state)
                buffer_average_p.append(average_policy)
                total_reward += reward
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)
                buffer_rho.append(1.0)
                self.memory.push(state, action, reward, policy, is_done)
                if t % args.t_max == 0 or is_done:
                    if is_done:
                        v_target = 0
                    else:
                        v_target = self.sess.run(self.v, {self.s: next_state[np.newaxis, :]})
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_target = r + args.gamma * v_target
                        buffer_v_target.append(v_target)
                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target, buffer_rho = \
                        np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target), np.vstack(buffer_rho)
                    feed_dict = {
                        self.s: buffer_s,
                        self.a_his: buffer_a,
                        self.target: buffer_v_target,
                        self.rho: buffer_rho,
                        self.average_p: buffer_average_p
                    }
                    # self.test()
                    # print(self.sess.run(self.a_gradients, feed_dict)[3])
                    self.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_average_p, buffer_rho = [], [], [], [], []
                    self.pull_global()
                    self.update_average()

                # Increment counters
                t += 1
                state = next_state
                if is_done:
                    ep_num += 1
                    print('Thread:', self.name, 'Reward:', total_reward)
                    break

        # call off policy part
        else:
            samples = self.memory.sample(args.batch_size)
            buffer_s = [m[0] for m in samples]
            buffer_a = [m[1] for m in samples]
            buffer_r = [m[2] for m in samples]
            buffer_p = [m[3] for m in samples]
            buffer_done = [m[4] for m in samples]
            # importance sampling
            buffer_rho, buffer_average_p, buffer_q_ret = [], [], []
            if buffer_done[-1]:
                q_ret = 0
            else:
                q_ret = self.sess.run(self.v, {self.s: buffer_s[-1][np.newaxis, :]})
            for i in reversed(range(len(samples))):
                pi = self.sess.run(self.a_prob, {self.s: buffer_s[i][np.newaxis, :]})[0][buffer_a[i]]
                mu = buffer_p[i][buffer_a[i]]
                rho_clipped = min(np.nan_to_num(pi / mu), args.c)
                average_p = average_net.get_policy(buffer_s[i])
                q_ret = buffer_r[i] + args.gamma * q_ret
                [buffer.append(item) for buffer, item in zip((buffer_rho, buffer_average_p, buffer_q_ret),
                                                             (rho_clipped, average_p, q_ret))]
                # q_i = self.sess.run(self.q, {self.s: buffer_s[i][np.newaxis, :]})[0][buffer_a[i]]
                # v_i = self.sess.run(self.v, {self.s: buffer_s[i][np.newaxis, :]})
                # # q_ret = rho_clipped * (q_ret - q_i) + v_i
            buffer_q_ret.reverse()
            buffer_rho.reverse()
            buffer_average_p.reverse()
            buffer_s, buffer_a, buffer_q_ret, buffer_rho = \
                np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_q_ret), np.vstack(buffer_rho)
            feed_dict = {
                self.s: buffer_s,
                self.a_his: buffer_a,
                self.target: buffer_q_ret,
                self.rho: buffer_rho,
                self.average_p: buffer_average_p
            }
            # self.test()
            # print(self.sess.run(self.a_gradients, feed_dict)[3])
            self.update_global(feed_dict)
            self.pull_global()
            self.update_average()

    # poisson random number by Knuth
    @staticmethod
    def _get_poisson(r):
        l, k, p = math.exp(-r), 0, 1
        while True:
            k += 1
            p *= random.uniform(0, 1)
            if not p > l:
                break
        return k - 1

    @staticmethod
    def _kl_div(p, q):
        return tf.reduce_sum(tf.multiply(p, tf.log(tf.clip_by_value(tf.div(p, q), 1e-20, 1.0))))



