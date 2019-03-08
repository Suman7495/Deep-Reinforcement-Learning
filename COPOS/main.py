import argparse
import gym, gym.spaces
from copos import *
import tensorflow as tf


def main():
    """
        Main script
        Default environment: CartPole-v0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gamma', type=float, default=.995)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    with tf.Session() as sess:
        agent = COPOS(args, sess)
        print("Training agent...\n")
        agent.train()
        print("Training completed.\nPrinting results")
        agent.print_results()

if __name__ == '__main__':
    main()
