import random
from Agent import Agent
from time import sleep
import cv2 as cv, numpy as np
import time, sys, csv
from datetime import datetime
from tqdm import tqdm
from Results import Results
import os

class Learning(object):
    """ docstring for Learning """
    def __init__(self, number_of_actions, input_dimension, load,
                    batch_size=25, episodes = 10,
                    max_steps = 100, epsilon = 0,
                    gamma = 0.0, alpha = 0.0,
                    epsilon_decay = 1.0, episodes_decay=30,
                    epochs = 1):
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.episodes_decay = episodes_decay
        self.epochs = epochs
        self.agent = Agent(number_of_actions, input_dimension, batch_size, self.alpha, load)
        self.heat_map = {'ACT0':0,'ACT1':0,'ACT2':0,'ACT3':0,'ACT4':0,'ACT5':0,'ACT6':0,'ACT7':0,'ACT8':0,'ACT9':0,'ACT10':0,'ACT11':0,'ACT12':0,'ACT13':0}
        self.analyzer = Results()

    """ append a new action in the memory, in form of a tuple, for further replay with a batch """
    def write_memory(self, memory, state, action, reward, next_state, is_done):
        memory.append((state, action, reward, next_state, is_done))
        #self.heat_map["ACT" + str(action)] += 1

    """ replays the memory in a batch, learning from past actions to maximize reward """
    def replay(self, state):

        mini_batch = random.sample(self.agent.memory, int(self.agent.batch_size))

        fit = None
        for state, action, reward, next_state, done in tqdm(mini_batch):
            target = reward
            if not done:
                target = (reward + self.gamma*(np.amax(self.agent.model.predict(next_state[1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1))[0])))

            target_f = self.agent.model.predict(state[1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1))
            target_f[0][action] = target
            fit = self.agent.model.fit(state[1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1), target_f, self.epochs, verbose=0)

        if fit == None:
            return 0
        else:
            return fit

    """ main loop for the learning itself """
    def run(self):
        for episode in range(self.episodes):
            self.agent.controller.start_sim()
            sleep(4)
            now = datetime.now()
            print str(now) + " starting ep " + str(episode+1)
            init = time.time()

            self.agent.instant_reward = 0.0
            state =  self.agent.vision.get_image(2) #state = (resolution, grayscale, colored RGB)

            steps_done = None
            for step in tqdm(range(self.max_steps)):
                steps_done = step

                cv.imshow("state", state[2])
                cv.waitKey(1)

                action_taken = self.agent.act(state[1], self.epsilon)
                next_state, reward, done = self.agent.do_step(action_taken) ##extrair imagem aqui dentro
                self.agent.instant_reward += reward
                self.write_memory(self.agent.memory, state, action_taken, reward, next_state, done)
                state = next_state
                if done:
                    break

            self.analyzer.steps_list.append(step+1)


            end = time.time()
            self.agent.controller.stop_sim()
            sleep(1)

            evall = None
            if len(self.agent.memory) > int(self.agent.batch_size):
                rep_init = time.time()
                evall = self.replay(state[1])
                rep_end = time.time()
                now = datetime.now()
                self.analyzer.mse_values.append(evall.history['mean_squared_error'])
                print str(now) + " mse value: ", str(round(evall.history['mean_squared_error'][0], 2)), " replay ", str(round((rep_end - rep_init)/60.0,2)), "minutes"

            self.analyzer.rewards_list.append(self.agent.instant_reward)
            self.agent.cummulative_reward += self.agent.instant_reward

            if  episode > 0 and (episode % self.episodes_decay == 0):
                self.epsilon *= self.epsilon_decay
                now = datetime.now()
                #print str(now) + " epsilon decay"

            if episode > 0 and episode%10 == 0:
                now = datetime.now()
                #print str(now) + " weights backup..."
                self.agent.model.save_weights('model_weights.h5')

            now = datetime.now()
            print (str(now) + " duration " + str(round((end - init)/60.0, 2)) +
                              " min //  ep " + str(episode+1) +
                              " // reward " + str(round(self.agent.instant_reward, 2)))

            self.agent.step_lost_counter = 0


        self.agent.controller.stop_sim()
        self.agent.controller.close_connection()
        cv.destroyAllWindows()

        now = datetime.now()
        #print str(now) + " saving model..."
        self.agent.model.save_weights('model_weights.h5')

        #print("------------------------ HEAT MAP ACTIONS -------------------")
        #print(self.heat_map)

        now = datetime.now()

        os.chdir("logs")
        os.mkdir(str(now))
        self.analyzer.plot_raw(self.analyzer.rewards_list, self.analyzer.reward_fig, str(now), "Reward x Episodio", "Reward")
        self.analyzer.plot_raw(self.analyzer.steps_list, self.analyzer.steps_fig, str(now), "Steps Gastos x Episodio", "Steps")
        self.analyzer.plot_raw(self.analyzer.mse_values, self.analyzer.mse_fig, str(now), "Mean Squared Error x Episodio", "Valor MSE", normalize=True)
        
