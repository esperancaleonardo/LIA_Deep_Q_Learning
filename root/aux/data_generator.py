import sys
sys.path.append("..")

from Vision import Vision
from Controller import Controller
from Agent import Agent
from tqdm import tqdm
import cv2 as cv, numpy as np
from time import sleep, time
import csv

number_of_samples = 10000
Agent = Agent(number_of_actions=12, input_dimension=299, batch_size=0, alpha=0, load=0, file_name = '')
#Agent.controller.connect(port=19996)
data_file = open('/media/leonardo/Seagate Expansion Driver/DTASET_IC/data_values.csv','w')
writer = csv.writer(data_file, delimiter=',')

Agent.controller.start_sim()


print("Generating " + str(number_of_samples) + " samples for training")

for sample in tqdm(range(number_of_samples)):
    initial_pos = [np.random.randint(0,360) for i in range(6)]
    Agent.controller.set_positions(Agent.handlers, initial_pos)

    sleep(0.5)

    initial_states = []
    initial_states.append(Agent.vision.get_image(sensor_number=1))
    initial_states.append(Agent.vision.get_image(sensor_number=2))
    initial_states.append(Agent.vision.get_image(sensor_number=3))

    action = Agent.act(initial_states[0], initial_states[1], initial_states[2], epsilon=11)
    new_state1, new_state2, new_state3, reward, done = Agent.do_step(action)
    new_states = [new_state1, new_state2, new_state3]

    writer.writerow([action, reward, done])

    for image in range(len(new_states)):
        cv.imwrite("/media/leonardo/Seagate Expansion Driver/DTASET_IC/init/"+str(sample+1)+"_"+str(image+1)+".png", initial_states[image][1])
        cv.imwrite("/media/leonardo/Seagate Expansion Driver/DTASET_IC/end/"+str(sample+1)+"_"+str(image+1)+".png", new_states[image][1])


data_file.close()
