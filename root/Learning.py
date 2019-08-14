import random
from Agent import Agent
from time import sleep
import cv2 as cv, numpy as np
import time, sys, csv
from datetime import datetime

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
        self.csv_file = open("csv_output_log.csv", 'wa')
        self.csv_writer = csv.writer(self.csv_file, delimiter = ',')
        self.csv_writer.writerow(["Episode", "Steps Done", "Steps Lost", "MSE", "Done Counter", "Epsilon", "Instant Reward", "Cummulative Reward"])
        self.heat_map = {'ACT0':0,'ACT1':0,'ACT2':0,'ACT3':0,'ACT4':0,'ACT5':0,'ACT6':0,'ACT7':0,'ACT8':0,'ACT9':0,'ACT10':0,'ACT11':0,'ACT12':0,'ACT13':0}

    """ append a new action in the memory, in form of a tuple, for further replay with a batch """
    def write_memory(self, memory, state, action, reward, next_state, is_done):
        memory.append((state, action, reward, next_state, is_done))
        self.heat_map["ACT" + str(action)] += 1

    """ replays the memory in a batch, learning from past actions to maximize reward """
    def replay(self, state):
        mini_batch = random.sample(self.agent.memory, int(self.agent.batch_size))

	fit = None
        for state, action, reward, next_state, done in mini_batch:
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
        run_init = time.time()

        for episode in range(self.episodes):
            self.agent.controller.start_sim()
            sleep(4)
            now = datetime.now()
            print str(now) + " starting ep " + str(episode+1)
            init = time.time()

            self.agent.instant_reward = 0.0
            state =  self.agent.vision.get_image(3) #state = (resolution, grayscale, colored RGB)

            steps_done = None
            for step in range(self.max_steps):
                steps_done = step
		
                action_taken = self.agent.act(state[1], self.epsilon)
                next_state, reward, done = self.agent.do_step(action_taken) ##extrair imagem aqui dentro
                self.agent.instant_reward = self.agent.instant_reward + reward

                self.write_memory(self.agent.memory, state, action_taken, reward, next_state, done)
                state = next_state
                if done:
                    now = datetime.now()
                    break

            end = time.time()
            self.agent.controller.stop_sim()
            sleep(2)

            evall = None
            if len(self.agent.memory) > self.agent.batch_size:
                rep_init = time.time()
                evall = self.replay(state[1])
                rep_end = time.time()
                now = datetime.now()
                print(str(now) + "mse value: ", evall.history['mean_squared_error'])
                print str(now) + " replay ", str((rep_end - rep_init)/60.0), "minutes"

            self.agent.cummulative_reward +=  + self.agent.instant_reward

            if evall != None:
                self.csv_writer.writerow([episode, steps_done, self.agent.step_lost_counter, evall.history['mean_squared_error'][0], self.agent.done_counter, self.epsilon, self.agent.instant_reward, self.agent.cummulative_reward])
            else:
                self.csv_writer.writerow([episode, steps_done, self.agent.step_lost_counter, 0, self.agent.done_counter, self.epsilon, self.agent.instant_reward, self.agent.cummulative_reward])


            if  episode > 0 and (episode % self.episodes_decay == 0):
                self.epsilon *= self.epsilon_decay
                now = datetime.now()
                print str(now) + " epsilon decay"

            if episode > 0 and episode%10 == 0:
                now = datetime.now()
                print str(now) + " weights backup..."
                self.agent.model.save_weights('model_weights.h5')

            now = datetime.now()
            print (str(now) + " duration (m) " + str((end - init)/60.0) +
                              " ep " + str(episode+1) +
                              " epsilon " + str(self.epsilon) +
                              " ep reward " + str(self.agent.instant_reward) +
                              " total reward " + str(self.agent.cummulative_reward) +
                              " times done " + str(self.agent.done_counter) +
                              " steps lost " + str(self.agent.step_lost_counter))
            run_stop = time.time()
            now = datetime.now()
            self.agent.step_lost_counter = 0


        self.agent.controller.stop_sim()
        self.agent.controller.close_connection()

        run_stop = time.time()
        now = datetime.now()
        print str(now) + " running for... " + str((run_stop - run_init)/60.0) + " minutes."

        now = datetime.now()
        print str(now) + " saving model..."
        self.agent.model.save_weights('model_weights.h5')

        print("------------------------ HEAT MAP ACTIONS -------------------")
        print(self.heat_map)

        self.csv_file.close()
