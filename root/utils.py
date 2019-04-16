import cv2 as cv


def bbPlotter(image, bbContours):

    for bb in bbContours:
        cv.rectangle(image,(bb[0]-15,bb[1]-15), (bb[0]+15,bb[1]+15), (0x00,0x00,0xff), 1)

    return image
