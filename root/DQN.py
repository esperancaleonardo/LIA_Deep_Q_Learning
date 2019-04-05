from Agent import Agent
from Learning import Learning
import Controller
import utils
import Vision
import sys
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# testar
#
#
#
#
#
#

def main():
    number_of_actions = 14
    input_dimension = 256
    batch_size=500
    episodes= int(sys.argv[2])
    max_steps=300
    epsilon=8
    gamma=0.1
    alpha=0.001
    epsilon_decay=0.95
    epochs=1
    load = int(sys.argv[1])

    dqn = Learning(number_of_actions, input_dimension, load, batch_size, episodes, max_steps, epsilon, gamma, alpha, epsilon_decay, epochs)

    #print dqn.episodes

    dqn.run()


if __name__ == '__main__':
    main()
