from Agent import Agent
import cv2 as cv
import tensorflow as tf
from tqdm import tqdm
import csv, os, sys

sys.dont_write_bytecode = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


Agent = Agent(12, 299, 1000, alpha = 0.006, load = 0, file_name = 'model_weights_train.h5')

csv_config = open("data_values.csv", 'r')
csv_reader = csv.reader(csv_config, delimiter = ',')
gamma = 0.99

rows = []
for row in csv_reader:
    rows.append(row)


for i in range(100):
    sample = 1

    for i in tqdm(range(len(rows))):
        action, reward, done = rows[i][0], rows[i][1], rows[i][2]
        initial_states = []
        initial_states.append(cv.cvtColor(cv.imread("dataset/init/"+str(sample)+"_"+str(1)+".png"), cv.IMREAD_GRAYSCALE))
        initial_states.append(cv.cvtColor(cv.imread("dataset/init/"+str(sample)+"_"+str(2)+".png"), cv.IMREAD_GRAYSCALE))
        initial_states.append(cv.cvtColor(cv.imread("dataset/init/"+str(sample)+"_"+str(3)+".png"), cv.IMREAD_GRAYSCALE))

        #print(initial_states[0])

        final_states = []
        final_states.append(cv.cvtColor(cv.imread("dataset/end/"+str(sample)+"_"+str(1)+".png"), cv.IMREAD_GRAYSCALE))
        final_states.append(cv.cvtColor(cv.imread("dataset/end/"+str(sample)+"_"+str(2)+".png"), cv.IMREAD_GRAYSCALE))
        final_states.append(cv.cvtColor(cv.imread("dataset/end/"+str(sample)+"_"+str(3)+".png"), cv.IMREAD_GRAYSCALE))

        sample += 1

        target = reward
        if not done:
            target = (reward + gamma*(np.amax(Agent.model.predict(
                                               [final_states[0].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                                final_states[1].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                                final_states[2].reshape(1,Agent.input_dimension,Agent.input_dimension,1)])[0])))

        target_f = Agent.model.predict(
                                           [initial_states[0].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                            initial_states[1].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                            initial_states[2].reshape(1,Agent.input_dimension,Agent.input_dimension,1)])
        action = int(action)
        target_f[0][action] = target
        fit = Agent.model.fit([initial_states[0].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                    initial_states[1].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                    initial_states[2].reshape(1,Agent.input_dimension,Agent.input_dimension,1)],
                                    target_f, 1, verbose=0)


    print(str(round(fit.history['mean_squared_error'][0], 2)))


Agent.model.save_weights('model_weights_train.h5')
