import random
import tensorflow as tf
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
        self.analyzer = Results()

    """ append a new action in the memory, in form of a tuple, for further replay with a batch """
    def write_memory(self, memory, state_list, action, reward, next_state_list, is_done):
        memory.append((state_list, action, reward, next_state_list, is_done))

    """ replays the memory in a batch, learning from past actions to maximize reward """
    def replay(self):

        mini_batch = random.sample(self.agent.memory, int(self.agent.batch_size))

        fit = None
        for state_list, action, reward, next_state_list, done in tqdm(mini_batch):
            target = reward
            if not done:
                target = (reward + self.gamma*(np.amax(self.agent.model.predict(
                                                   [next_state_list[0][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1),
                                                    next_state_list[1][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1),
                                                    next_state_list[2][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1)])[0])))

            target_f = self.agent.model.predict(
                                               [state_list[0][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1),
                                                state_list[1][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1),
                                                state_list[2][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1)])
            target_f[0][action] = target
            fit = self.agent.model.fit([state_list[0][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1),
                                        state_list[1][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1),
                                        state_list[2][1].reshape(1,self.agent.input_dimension,self.agent.input_dimension,1)],
                                        target_f, self.epochs, verbose=0)

        if fit == None:
            return 0
        else:
            return fit

    """ main loop for the learning itself """
    def run(self):
        black = np.zeros((299,299,3), np.uint8)

        for episode in range(self.episodes):
            self.agent.controller.start_sim()
            sleep(4)
            now = datetime.now()
            print str(now) + " starting ep " + str(episode+1)
            init = time.time()
	
            state_list = []
            self.agent.instant_reward = 0.0
            state_list.append(self.agent.vision.get_image(1)) #state = (resolution, grayscale, colored RGB)
            state_list.append(self.agent.vision.get_image(2)) #state = (resolution, grayscale, colored RGB)
            state_list.append(self.agent.vision.get_image(3)) #state = (resolution, grayscale, colored RGB)

            steps_done = None
            for step in tqdm(range(self.max_steps)):
                steps_done = step

                #h1 = np.concatenate((state_list[0][2], state_list[1][2]), axis=1)
                #h2 = np.concatenate((state_list[2][2], black), axis=1)
                #full = np.concatenate((h1, h2), axis=0)
                #cv.imshow("state", state_list[0][2])
                #cv.waitKey(1)

                action_taken = self.agent.act(state_list[0], state_list[1], state_list[2], self.epsilon)
                next_state1, next_state2, next_state3, reward, done = self.agent.do_step(action_taken) ##extrair imagem aqui dentro
                self.agent.instant_reward += reward
                self.write_memory(self.agent.memory, state_list, action_taken, reward, [next_state1, next_state2, next_state3], done)
                state_list[0] = next_state1
                state_list[1] = next_state2
                state_list[2] = next_state3

                if done:
                    break

            self.analyzer.steps_list.append(step+1)


            end = time.time()
            self.agent.controller.stop_sim()
            sleep(3)

            evall = None
            if len(self.agent.memory) > int(self.agent.batch_size):
                rep_init = time.time()
                evall = self.replay()
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
        self.agent.model.save_weights('model_weights.h5')


        now = datetime.now()

        os.chdir("logs")
	dirr = str(now)        
	os.mkdir(dirr)
	self.analyzer.plot_media_n(self.analyzer.rewards_list, self.analyzer.reward_fig, dirr, 10, "REWARDxEP", "Reward Media x 10 Episodio", "Reward Media")
        self.analyzer.plot_raw(self.analyzer.rewards_list, self.analyzer.reward_fig, dirr, "REWARDxEP", "Reward x Episodio", "Reward")
        self.analyzer.plot_raw(self.analyzer.steps_list, self.analyzer.steps_fig, dirr, "STEPS", "Steps Gastos x Episodio", "Steps")
        self.analyzer.plot_raw(self.analyzer.mse_values, self.analyzer.mse_fig, dirr, "MSE", "Mean Squared Error x Episodio", "Valor MSE", normalize=True)
