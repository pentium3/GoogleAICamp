import cv2
from matplotlib import pyplot as plt
import numpy as np

def task3mask(inputfile, outputfile):
    img=cv2.imread(inputfile)
    #img=cv2.imread('input/cartoon.jpg')
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_blue=np.array([0,0,221])
    upper_blue=np.array([180,30,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    erode=cv2.erode(mask,None,iterations=10)
    dilate=cv2.dilate(erode,None,iterations=10)
    #cv2.imwrite('input/maskcartoon.jpg',dilate)
    cv2.imwrite(outputfile, dilate)

inputfile='cartoon.jpg'
outputfile='cartoonmask.jpg'
task3mask(inputfile, outputfile)





