from source import vrep
from time import sleep
from PIL import Image as I
import array, cv2 as cv, numpy as np
import imutils, sys

sys.dont_write_bytecode = True


class Vision(object):
    """ docstring for Vision """
    def __init__(self, sensor_1, sensor_2, sensor_3, client_id):
        self.client_id = client_id

        _, self.sensor_1 = vrep.simxGetObjectHandle(self.client_id, sensor_1, vrep.simx_opmode_oneshot_wait)
        _, self.sensor_2 = vrep.simxGetObjectHandle(self.client_id, sensor_2, vrep.simx_opmode_oneshot_wait)
        _, self.sensor_3 = vrep.simxGetObjectHandle(self.client_id, sensor_3, vrep.simx_opmode_oneshot_wait)

        err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_1, 0, vrep.simx_opmode_streaming)
        err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_2, 0, vrep.simx_opmode_streaming)
        err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_3, 0, vrep.simx_opmode_streaming)

    # OK
    """ retorna a resolucao e a imagem obtida do sensor sensor_number do vrep como um array """
    def get_image(self, sensor_number, grayscale = True, upscale = False):
        if sensor_number == 1:
            err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_1, 0, vrep.simx_opmode_buffer)
        elif sensor_number == 2:
            err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_2, 0, vrep.simx_opmode_buffer)
        else:
            err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_3, 0, vrep.simx_opmode_buffer)

        image = array.array('b', image)
        image = I.frombuffer("RGB", (resolution[0],resolution[1]), image, "raw", "RGB", 0, 1)
        image = np.asarray(image)
        image = cv.flip(image, 0)

        if grayscale:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if upscale:
            image = cv.resize(image, (512,512))

        
        return resolution, gray, cv.cvtColor(image, cv.COLOR_BGR2RGB)


    # OK
    """ detecta e devolve (se houver) as coordenadas de um objeto em uma imagem
        dados seu range de cor em HSV """
    def track_collor(self, image, color):
        # Blur the image to reduce noise
        # Convert RGB to HSV
        hsv = cv.cvtColor(cv.GaussianBlur(image, (5,5),0), cv.COLOR_BGR2HSV)

        if color == 0: #red
            # Thresholds the HSV image
            lower = np.array([0,20,70])
            upper = np.array([10,255,255])
        else: #BLUE
            # Threshold the HSV image
            lower = np.array([11,50,50])
            upper = np.array([130,255,255])

        mask = cv.GaussianBlur(cv.inRange(hsv, lower, upper), (5,5), 0) # Blur the mask

        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours_list = []

        for cont in contours:
            moments = cv.moments(cont)
            m00 = moments['m00']
            if m00 != 0:
                centroid_x = int(moments['m10']/m00)
                centroid_y = int(moments['m01']/m00)

                if centroid_x != None and centroid_y != None:
                    contours_list.append((centroid_x, centroid_y))

        return contours_list
