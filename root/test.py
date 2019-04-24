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

