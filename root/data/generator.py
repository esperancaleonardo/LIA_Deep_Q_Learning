import cv2 as cv
import os
import sys
sys.path.append("..")
from Controller import *
from Vision import *
from time import sleep
from PIL import Image as I
import array, numpy as np
import imutils
from random import randint

n_samples = 1

print "acquiring " + str(n_samples) + " samples."
path = os.getcwd()
meu_controller = Controller("UR3", 6)
meu_controller.connect(19996)

handler_strings = []
for i in range(0,6):
    handler_strings.append("_joint" + str(i+1))

handlers = meu_controller.get_handlers(handler_strings)

#print handlers


vision = Vision('Vision_frontal','Vision_lateral','Vision_top',meu_controller.id_number)

meu_controller.start_sim()

sleep(4)
print "warming up"
print path

black = np.zeros((299,299,3), np.uint8)

for cont in range(n_samples):
    print cont,
    res, _, image1 = vision.get_image(1, grayscale = True, upscale = False)
    # cv.imwrite(os.path.join(path + '/img1_' + str(cont) + '.jpg'), image1)
    sleep(1)
    res, _, image2 = vision.get_image(2, grayscale = True, upscale = False)
    # cv.imwrite(os.path.join(path + '/img2_' + str(cont) + '.jpg'), image2)
    sleep(1)
    res, _, image3 = vision.get_image(3, grayscale = True, upscale = False)
    # cv.imwrite(os.path.join(path + '/img3_' + str(cont) + '.jpg'), image3)


    h1 = np.concatenate((image1, image2), axis=1)
    h2 = np.concatenate((image3, black), axis=1)
    full = np.concatenate((h1, h2), axis=0)

    cv.imwrite(os.path.join(path + '/full_state299.jpg'), cv.resize(full,(299,299)))

    sleep(1)
    #
    # meu_controller.stop_sim()
    # sleep(1)
    # meu_controller.start_sim()
    #
    # j1 = randint(0,360)
    # j2 = randint(-90,90)
    # j3 = randint(-160,160)
    # j4 = randint(0,360)
    # j5 = randint(-120,120)
    # j6 = randint(0,360)
    # g  = randint(0,1)
    #
    #
    #
    # meu_controller.set_positions(handlers, [j1, j2, j3, j4, j5, j6])
    # if g == 0:
    #     meu_controller.gripper_close()
    # else:
    #     meu_controller.gripper_open()
    # sleep(5)


meu_controller.stop_sim()
meu_controller.close_connection()
