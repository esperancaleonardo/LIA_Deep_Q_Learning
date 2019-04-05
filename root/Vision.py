from source import vrep
from time import sleep
from PIL import Image as I
import array, cv2 as cv, numpy as np
import imutils

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
    """ retorna a resolucao e a imagem obtida do sensor 1 do vrep como um array """
    def get_image_1(self, grayscale = True, upscale = False):
        err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_1, 0, vrep.simx_opmode_buffer)

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
    """ retorna a resolucao e a imagem obtida do sensor 2 do vrep como um array """
    def get_image_2(self, grayscale = True, upscale = False):
        err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_2, 0, vrep.simx_opmode_buffer)

        image = array.array('b', image)
        image = I.frombuffer("RGB", (resolution[0],resolution[1]), image, "raw", "RGB", 0, 1)
        image = np.asarray(image)
        image = cv.flip(image, 0)

        if grayscale:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if upscale:
            image = cv.resize(image, (512,512))

        # cv.imshow("side", cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # cv.waitKey(1)

        return resolution, gray, cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # OK
    """ retorna a resolucao e a imagem obtida do sensor 3 do vrep como um array """
    def get_image_3(self, grayscale = True, upscale = False):
        err, resolution, image = vrep.simxGetVisionSensorImage(self.client_id, self.sensor_3, 0, vrep.simx_opmode_buffer)

        image = array.array('b', image)
        image = I.frombuffer("RGB", (resolution[0],resolution[1]), image, "raw", "RGB", 0, 1)
        image = np.array(image)

        if grayscale:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if upscale:
            image = cv.resize(image, (512,512))

        cv.imshow("top", cv.cvtColor(image, cv.COLOR_BGR2RGB))
        cv.waitKey(1)
        return resolution, gray, cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # OK
    """ detecta e devolve (se houver) as coordenadas de um objeto em uma imagem
        dados seu range de cor em HSV """
    def track_collor(self, image, color):

        blur = cv.GaussianBlur(image, (5,5),0) # Blur the image to reduce noise
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV) # Convert RGB to HSV

        if color == 0: #red
            # Thresholds the HSV image
            lower2 = np.array([0,20,70])
            upper2 = np.array([10,255,255])
            mask = cv.inRange(hsv, lower2, upper2) # Threshold the HSV image

        else: #BLUE
            lower1 = np.array([11,50,50])
            upper1 = np.array([130,255,255])
            mask = cv.inRange(hsv, lower1, upper1)

        bmask = cv.GaussianBlur(mask, (5,5),0) # Blur the mask



        contours = cv.findContours(bmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        contours_list = []

        for cont in contours:
            moments = cv.moments(cont)
            m00 = moments['m00']
            centroid_x, centroid_y = None, None
            if m00 != 0:
                centroid_x = int(moments['m10']/m00)
                centroid_y = int(moments['m01']/m00)

            ctr = None
            if centroid_x != None and centroid_y != None:
                ctr = (centroid_x, centroid_y)
                contours_list.append(ctr)

        return contours_list
