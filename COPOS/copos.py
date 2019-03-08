import gym, gym.spaces
import tensorflow as tf
import numpy as np
import sys

from .hyperameters import *
from .solver import *

def main(env_name='CartPole-v0, seed=1, run_name=None):
    """
        Main script to run training
    """
    # Make gym environment
    env = gym.make(env_name)

    # Initialize seeds
    seed = init(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    # Configure Tensorflow
    # Finish this section

    # Initialize placeholders
    obs_size = env.observation.space()

if __name__ == '__main__'
    if len(sys.argv) > 1:
        main(sys.argv)
    else
        raise Exception('Missing Environment')
