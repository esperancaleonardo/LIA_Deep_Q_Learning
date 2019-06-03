from Agent import Agent
from Learning import Learning
import Controller
import Vision
import sys
import os
import tensorflow as tf
import csv



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    number_of_actions = 12
    input_dimension = 128

    episodes= int(sys.argv[2])
    load = int(sys.argv[1])

    csv_config = open("config.csv", 'r')
    csv_reader = csv.reader(csv_config, delimiter = ',')
    file = {}
    for row in csv_reader:
        file[row[0]] = row[1]

    #print file
    csv_config.close()

    max_steps=int(file.get('max_steps'))
    epsilon=float(file.get('epsilon'))
    batch_size=int(file.get('batch_size'))
    epochs=int(file.get('epochs'))
    epsilon_decay=float(file.get('epsilon_decay'))
    episodes_decay=int(file.get('episodes_decay'))
    alpha=float(file.get('alpha'))
    gamma=float(file.get('gamma'))

    dqn = Learning( number_of_actions,
                    input_dimension,
                    load,
                    batch_size,
                    episodes,
                    max_steps,
                    epsilon,
                    gamma,
                    alpha,
                    epsilon_decay,
                    episodes_decay,
                    epochs
                )
    dqn.agent.cummulative_reward = float(file.get('cummulative_reward'))

    dqn.run()

    file = {'epsilon':dqn.epsilon,
            'cummulative_reward':dqn.agent.cummulative_reward,
            'max_steps':dqn.max_steps,
            'batch_size':dqn.agent.batch_size,
            'epochs':dqn.epochs,
            'epsilon_decay':dqn.epsilon_decay,
            'episodes_decay':dqn.episodes_decay,
            'alpha':dqn.alpha,
            'gamma':dqn.gamma}


    csv_config = open("config.csv", 'w')
    csv_writer = csv.writer(csv_config, delimiter = ',')

    for item in file:
        csv_writer.writerow([str(item), file.get(item)])
    csv_config.close()

if __name__ == '__main__':
    main()
