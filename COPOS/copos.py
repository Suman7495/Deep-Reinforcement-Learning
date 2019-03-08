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
        self.render = False

        # Hyperparameters
        self.lr = args.lr
        self.gamma = args.gamma
        self.num_ep = args.num_ep

        # Initialize empty reward list
        self.rew_list = []

        # Build policy model
        self.model = self.build_model()


    def build_model(self):
        """
            Neural Network Model of the COPOS agent
        """
        return

    def pick_action(self):
        """
            Choose an action
        """
        return

    def train(self):
        """
            Train using COPOS algorithm
        """
        return

    def print_results(self):
        """
            Plot the results
        """
        return