import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from Vision import Vision
from Controller import Controller
from collections import deque
from datetime import datetime
import cv2 as cv, os
import numpy as np
from time import sleep
import math
import sys

sys.path.append("..")
path = os.getcwd()

class Agent(object):
    """ docstring for Agent """
    def __init__(self, number_of_actions, input_dimension, batch_size, alpha , load):
        self.number_of_actions = number_of_actions
        self.input_dimension = input_dimension
        self.instant_reward = 0.0
        self.cummulative_reward = 0.0
        self.memory = deque(maxlen=100000)
        self.batch_size = int (batch_size)
        self.classes = self.number_of_actions
        self.controller = Controller("UR3", 6)
        self.controller.connect(19996)
        self.vision = Vision('Vision_frontal','Vision_lateral','Vision_top',self.controller.id_number)
        self.model = self.create_model(input_dimension, number_of_actions, 'mean_squared_error', Adam(lr=alpha),  ['accuracy', 'mean_squared_error'])
        if load == 1:   #load previous weights if set to 1
            self.model.load_weights('model_weights.h5')
            now = datetime.now()
            print str(now) + " model weights load done!"
        self.handlers = self.manage_handlers()
        self.counter = 0
        self.step_degrees = 45.0
        self.done_counter = 0
        self.step_lost_counter = 0

    """ manage handlers for action manipulation """
    def manage_handlers(self):
        handler_strings = []
        for i in range(0,6):
            handler_strings.append("_joint" + str(i+1))

        handlers = self.controller.get_handlers(handler_strings)
        return handlers

    """ creates the DQN model with a specified input dimension and a number of actions for the output layers """
    def create_model(self, input_dimension, number_of_actions, loss_type, optimizer, metrics_list):
        model1 = Sequential()
        model2 = Sequential()
        model3 = Sequential()

        model1.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Conv2D(64, (5, 5), activation='relu'))
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Conv2D(64, (5, 5), activation='relu'))
        model1.add(MaxPooling2D(pool_size=(2, 2)))
        model1.add(Dropout(0.25))
        model1.add(Flatten())
        ###################################################################################################
        model2.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
        model2.add(MaxPooling2D(pool_size=(2, 2)))
        model2.add(Conv2D(64, (5, 5), activation='relu'))
        model2.add(MaxPooling2D(pool_size=(2, 2)))
        model2.add(Conv2D(64, (5, 5), activation='relu'))
        model2.add(MaxPooling2D(pool_size=(2, 2)))
        model2.add(Dropout(0.25))
        model2.add(Flatten())
        ###################################################################################################
        model3.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
        model3.add(MaxPooling2D(pool_size=(2, 2)))
        model3.add(Conv2D(64, (5, 5), activation='relu'))
        model3.add(MaxPooling2D(pool_size=(2, 2)))
        model3.add(Conv2D(64, (5, 5), activation='relu'))
        model3.add(MaxPooling2D(pool_size=(2, 2)))
        model3.add(Dropout(0.25))
        model3.add(Flatten())

        combined = concatenate([model1.output, model2.output, model3.output])
        x = Dense(4096)(combined)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(number_of_actions)(x)

        model = Model(inputs=[model1.input, model2.input, model3.input], output=x)
        model.compile(loss = loss_type, optimizer = optimizer, metrics = metrics_list)

        # model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(input_dimension,input_dimension, 1)))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(64, (5, 5), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(64, (5, 5), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(4096))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(number_of_actions))
        # model.compile(loss = loss_type, optimizer = optimizer, metrics = metrics_list)
        # #model.summary()

        return model

    """ decides to act randomly or not, aconding to epsilon value """
    def act(self, state1, state2, state3, epsilon):

        if self.step_lost_counter < 30:
            if np.random.randint(0,10) <= epsilon:
                return np.random.randint(0,self.number_of_actions)
            else:
                state1 = np.array(state1)
                state2 = np.array(state2)
                state3 = np.array(state3)
                action_values = self.model.predict(
                                                   [state1.reshape(1,self.input_dimension,self.input_dimension,1),
                                                    state2.reshape(1,self.input_dimension,self.input_dimension,1),
                                                    state3.reshape(1,self.input_dimension,self.input_dimension,1)]
                                                  )
                return np.argmax(action_values[0]) ## check if correct
        else:
            self.step_lost_counter = 0
            return np.random.randint(0,self.number_of_actions)


    """ moves with a certain action, moving one joint 20 degrees clockwise or counter clockwise """
    def do_step(self, action):
        #joint 1
        if action == 0:
            self.controller.set_joint_position(self.handlers[0], self.controller.get_joint_position(self.handlers[0]) + self.step_degrees)
        elif action == 1:
            self.controller.set_joint_position(self.handlers[0], self.controller.get_joint_position(self.handlers[0]) - self.step_degrees)
        #joint 2
        elif action == 2:
            self.controller.set_joint_position(self.handlers[1], self.controller.get_joint_position(self.handlers[1]) + self.step_degrees)
        elif action == 3:
            self.controller.set_joint_position(self.handlers[1], self.controller.get_joint_position(self.handlers[1]) - self.step_degrees)
        #joint 3
        elif action == 4:
            self.controller.set_joint_position(self.handlers[2], self.controller.get_joint_position(self.handlers[2]) + self.step_degrees)
        elif action == 5:
            self.controller.set_joint_position(self.handlers[2], self.controller.get_joint_position(self.handlers[2]) - self.step_degrees)
        #joint 4
        elif action == 6:
            self.controller.set_joint_position(self.handlers[3], self.controller.get_joint_position(self.handlers[3]) + self.step_degrees)
        elif action == 7:
            self.controller.set_joint_position(self.handlers[3], self.controller.get_joint_position(self.handlers[3]) - self.step_degrees)
        #joint 5
        elif action == 8:
            self.controller.set_joint_position(self.handlers[4], self.controller.get_joint_position(self.handlers[4]) + self.step_degrees)
        elif action == 9:
            self.controller.set_joint_position(self.handlers[4], self.controller.get_joint_position(self.handlers[4]) - self.step_degrees)
        #joint 6
        elif action == 10:
            self.controller.set_joint_position(self.handlers[5], self.controller.get_joint_position(self.handlers[5]) + self.step_degrees)
        else: #if action == 11:
            self.controller.set_joint_position(self.handlers[5], self.controller.get_joint_position(self.handlers[5]) - self.step_degrees)
        # # gripper
        # elif action == 12:
        #     self.controller.gripper_open()
        # else: #action == 13:
        #     self.controller.gripper_close()

        sleep(0.3)
        new_state1, new_state2, new_state3, reward,  done = self.get_reward()

        return new_state1, new_state2, new_state3, reward, done

    """ etimates a reward value using computer vision for 3d distance between two objects """
    def get_reward(self):

        new_state1 = self.vision.get_image(1)   #frontal
        new_state2 = self.vision.get_image(2)   #lateral
        new_state3 = self.vision.get_image(3)   #top

        red_centers1 = self.vision.track_collor(new_state3[2], 0)
        red_centers2 = self.vision.track_collor(new_state2[2], 0)
        blue_center1 = self.vision.track_collor(new_state3[2], 1)
        blue_center2 = self.vision.track_collor(new_state2[2], 1)

        if (len(blue_center1) > 0) and (len(blue_center2) > 0) and (len(red_centers1) > 0) and (len(red_centers2) > 0):
            red1 = red_centers1[0]
            red2 = red_centers2[0]
            blue1 = blue_center1[0]
            blue2 = blue_center2[0]

            _3d_red = (red1[0],red1[1],red2[1])
            _3d_blue = (blue1[0],blue1[1],blue2[1])

            #print(_3d_red, _3d_blue)

            distance = math.sqrt((_3d_red[0]-_3d_blue[0])**2 + (_3d_red[1]-_3d_blue[1])**2 + (_3d_red[2]-_3d_blue[2])**2)
            base = 150

            if distance >= 11.0:
                done = 0
                reward = -20*distance if distance >= 50.0 else base - 2*distance
            else:
                self.done_counter +=1
                done = 1
                reward = 10*base
        else:
            self.step_lost_counter +=1
            reward = 0.0
            done = 0

        return new_state1, new_state2, new_state3, reward, done
