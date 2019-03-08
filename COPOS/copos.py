import gym, gym.spaces
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


class COPOS:
    def __init__(self, args, sess):
        """
        Initialize COPOS agent class
        """
        env = gym.make(args.env)
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        self.render = True

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.num_ep = args.num_ep

        # Initialize empty reward list
        self.rew_list = []

        # Build policy model
        self.__init__placeholders()
        self.build_policy()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def __init__placeholders(self):
        """
            Define Tensorflow placeholders
        """
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
        self.act = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='act')
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None], name='adv')
        self.old_std = tf.placeholder(dtype=tf.float32, shape=[None, self.act_dim], name='old_std')
        self.old_mean = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='old_mean')

    def build_policy(self):
        """
            Neural Network Model of the COPOS agent
        """
        # Create the neural network with output as mean
        self.mean = [1., -1]
        self.std = [1, 2.]
        self.act_dist = tfp.distributions.MultivariateNormalDiag(self.mean, self.std)

    def pick_action(self, state):
        """
            Choose an action
        """
        action = self.act_dist.sample().eval()
        return np.argmax(action)

    def store_memory(self, transition):
        return

    def loss(self):
        """
            Compute loss
        """
        # Log probabilities of new and old actions
        old_act_dist = tfp.distributions.MultivariateNormalDiag(self.old_mean, self.old_std)
        self.old_log_prob = tf.reduce_sum(old_act_dist.log_prob(self.act))
        self.log_prob = tf.reduce_sum(self.act_dist.log_prob(self.act))
        prob_ratio = tf.exp(self.log_prob - self.old_log_prob)

        # Surrogate Loss
        self.surrogate_loss = -tf.reduce_mean(prob_ratio*self.adv)

        # KL Divergence
        self.kl = tfp.distributions.kl_divergence(self.act_dist, old_act_dist)

        # Entropy
        self.entropy = tf.reduce_mean(self.act_dist.entropy())

    def train(self):
        """
            Train using COPOS algorithm
        """
        for ep in range(int(self.num_ep)):
            state = self.env.reset()
            state = np.reshape(state, [1, self.obs_dim])
            tot_rew = 0
            done = False
            t = 0
            # Iterate over timesteps
            while t < 1000:
                t += 1
                if self.render:
                    self.env.render()
                act = self.pick_action(state)

                # Get next state and reward from environment
                next_state, rew, done, info = self.env.step(act)
                next_state = np.reshape(next_state, [1, self.obs_dim])
                # Store transition in memory
                transition = deque((state, act, rew, next_state, done))
                self.store_memory(transition)
                tot_rew += rew
                state = next_state
                if done:
                    print("\nEpisode: {}/{}, score: {}"
                          .format(ep, self.num_ep, t))
                    break

            self.rew_list.append(tot_rew)
            # Compute loss, yet to finish

    def print_results(self):
        """
            Plot the results
        """
        return