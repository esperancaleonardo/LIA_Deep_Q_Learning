from source import vrep
from Vision import Vision
from Controller import Controller
from time import sleep
import math

cont = Controller("UR3", 6)
cont.connect()


vision = Vision('Vision_frontal','Vision_lateral','Vision_top',cont.id_number)

cont.start_sim()

sleep(1)

while(True):

    new_state = vision.get_image_3()

    aux_state = vision.get_image_2()

    colored = new_state[2]
    aux_colored = aux_state[2]

    red_centers1 = vision.track_collor(colored, 0)
    red_centers2 = vision.track_collor(aux_colored, 0)

    blue_center1 = vision.track_collor(colored, 1)
    blue_center2 = vision.track_collor(aux_colored, 1)

    red1 = red_centers1[0]
    red2 = red_centers2[0]

    blue1 = blue_center1[0]
    blue2 = blue_center2[0]

    #print red1, red2, blue1, blue2

    _3d_red = (red1[0],red1[1],red2[1])
    _3d_blue = (blue1[0],blue1[1],blue2[1])
    #print _3d_red, _3d_blue

    distance = math.sqrt((_3d_red[0]-_3d_blue[0])**2 + (_3d_red[1]-_3d_blue[1])**2 + (_3d_red[2]-_3d_blue[2])**2)
    print distance



cont.close_connection()
