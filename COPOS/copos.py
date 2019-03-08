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
        self.model = self.policy()


    def policy(self):
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
            Train using DQN algorithm
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
                #self.store_memory(transition)
                tot_rew += rew
                state = next_state
                if done:
                    print("\nEpisode: {}/{}, score: {}"
                          .format(ep, self.num_ep, t))
                    break
                if len(self.memory) > self.batch_size:
                    self.replay()

            self.rew_list.append(tot_rew)

    def print_results(self):
        """
            Plot the results
        """
        return