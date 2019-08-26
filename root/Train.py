from Agent import Agent
import cv2 as cv
from tqdm import tqdm

Agent = Agent(12, 299, 1000, alpha = _alpha, load = 1)

csv_config = open("data_values.csv", 'r')
csv_reader = csv.reader(csv_config, delimiter = ',')
gamma = 0.99

rows = []
for row in csv_reader:
    rows.append(row)

sample = 1

for row in twdm(len(rows)):
    reward, done = row[0], row[1]
    initial_states = []
    initial_states.append(cv.imread("dataset/init/"+str(sample)+"_"+str(1)+".png"))
    initial_states.append(cv.imread("dataset/init/"+str(sample)+"_"+str(2)+".png"))
    initial_states.append(cv.imread("dataset/init/"+str(sample)+"_"+str(3)+".png"))

    final_states = []
    final_states.append(cv.imread("dataset/end/"+str(sample)+"_"+str(1)+".png"))
    final_states.append(cv.imread("dataset/end/"+str(sample)+"_"+str(2)+".png"))
    final_states.append(cv.imread("dataset/end/"+str(sample)+"_"+str(3)+".png"))

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
    target_f[0][action] = target
    fit = Agent.model.fit([initial_states[0].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                initial_states[1].reshape(1,Agent.input_dimension,Agent.input_dimension,1),
                                initial_states[2].reshape(1,Agent.input_dimension,Agent.input_dimension,1)],
                                target_f, 1, verbose=0)

    print(str(round(fit.history['mean_squared_error'][0], 2)))

Agent.model.save_weights('model_weights.h5')
