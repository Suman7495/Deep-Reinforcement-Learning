import gym, gym.spaces
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


class COPOS:
    def __init__(self, args, sess):
        """
        Initialize COPOS agent class
        """
        env = gym.make(args.env)
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

    def build_model(self):
        return

    def pick_action(self):
        return

    def train(self):
        return

    def print_results(self):
        return