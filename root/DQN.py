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

#   ########## testar
#   max_steps 1,000
#   gamma (discount rate) 0.99
#   batch_size 20,000
#   memory size 500,000
#   alpha (learning rate) 0.000006
#
#

def main():
    number_of_actions = 14
    input_dimension = 128
    batch_size=50
    episodes= int(sys.argv[2])
    max_steps=201
    epsilon=4
    gamma=0.99
    alpha=0.000006
    epsilon_decay=0.95
    epochs=1
    load = int(sys.argv[1])

    dqn = Learning(number_of_actions, input_dimension, load, batch_size, episodes, max_steps, epsilon, gamma, alpha, epsilon_decay, epochs)

    #print dqn.episodes

    dqn.run()


if __name__ == '__main__':
    main()
