import random
from Agent import Agent
from time import sleep
import cv2 as cv
import time
import sys
import numpy as np
from datetime import datetime

class Learning(object):
    """ docstring for Learning """
    def __init__(self, number_of_actions, input_dimension, load, batch_size=25, episodes = 10, max_steps = 100, epsilon = 0, gamma = 0.0, alpha = 0.0, epsilon_decay = 1.0, episodes_decay=30, epochs = 1):
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.episodes_decay = episodes_decay
        self.epochs = epochs
        self.agent = Agent(number_of_actions, input_dimension, batch_size, self.alpha, load)



    def write_memory(self, memory, state, action, reward, next_state, is_done):
        memory.append((state, action, reward, next_state, is_done))


    def replay(self, state):
        mini_batch = random.sample(self.agent.memory, int(self.agent.batch_size))


        for state, action, reward, next_state, done in mini_batch:
            if not done:
                target = (reward + self.gamma*(np.amax(self.agent.model.predict(next_state[1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1))[0])))
            else:
                target = reward


            target_f = self.agent.model.predict(state[1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1))
            target_f[0][action] = target
            self.agent.model.fit(state[1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1), target_f, self.epochs, verbose=0)



    def run(self):
        self.agent.controller.start_sim()
        sleep(2)

        run_init = time.time()

        for episode in range(self.episodes):
            now = datetime.now()
            print str(now) + " starting ep " + str(episode+1)

            init = time.time()
            self.agent.instant_reward = 0.0
            state =  self.agent.vision.get_image_3() #state = (resolution, grayscale, colored RGB)


            for step in range(self.max_steps):

                action_taken = self.agent.act(state[1], self.epsilon)
                next_state, reward, done = self.agent.do_step(action_taken) ##extrair imagem aqui dentro
                self.agent.instant_reward = self.agent.instant_reward + reward

                self.write_memory(self.agent.memory, state, action_taken, reward, next_state, done)
                state = next_state
                if done:
                    now = datetime.now()
                    print str(now) + " DONE"
                    break

                #if step > 0 and step%50==0:
                #    now = datetime.now()
                #    print str(now) + " step ", step

            end = time.time()
            self.agent.controller.stop_sim()
            sleep(2)

            # TODO: inserir verificacao de quantidade de passos dados
            if len(self.agent.memory) > self.agent.batch_size:
                now = datetime.now()
                print str(now) + " REPLAY"
                rep_init = time.time()
                self.replay(state[1])
                rep_end = time.time()
                now = datetime.now()
                print str(now) + " replay ", str((rep_end - rep_init)/60.0), "minutes"

            self.agent.cummulative_reward = self.agent.cummulative_reward + self.agent.instant_reward

            if  episode > 0 and episode%self.episodes_decay == 0:
                self.epsilon = self.epsilon * self.epsilon_decay
                now = datetime.now()
                print str(now) + " epsilon decay"

            if episode > 0 and episode%10 == 0:
                now = datetime.now()
                print str(now) + " weights backup..."
                self.agent.model.save_weights('model_weights.h5')


            self.agent.controller.start_sim()
            sleep(4)

            now = datetime.now()
            print str(now) + " duration (m) " + str((end - init)/60.0) + " ep " + str(episode+1) + " epsilon " + str(self.epsilon) + " ep reward " + str(self.agent.instant_reward) + " total reward " + str(self.agent.cummulative_reward)
            run_stop = time.time()
            now = datetime.now()
            print str(now) + " running for... " + str((run_stop - run_init)/60.0) + " minutes."


        self.agent.controller.stop_sim()
        self.agent.controller.close_connection()

        run_stop = time.time()
        now = datetime.now()
        print str(now) + " running for... " + str((run_stop - run_init)/60.0) + " minutes."

        now = datetime.now()
        print str(now) + " saving model..."
        self.agent.model.save_weights('model_weights.h5')
