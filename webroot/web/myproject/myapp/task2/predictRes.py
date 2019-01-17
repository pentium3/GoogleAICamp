import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
from scipy.spatial.distance import cdist
with open("cartoon.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
input1 = [1,1,0,1,0,1,0,1,0,1]

allData = []
for i in range(len(content)):
    str = content[i]
    char = list(str)
    arr = []
    arr.append(char[1])
    arr.append(char[3])
    arr.append(char[5])
    arr.append(char[7])
    arr.append(char[9])
    arr.append(char[11])
    arr.append(char[13])
    arr.append(char[15])
    arr.append(char[17])
    arr.append(char[19])
    allData.append(arr)

similar = 0
input = []
input.append(input1)
dis = 999999999
for i in range(len(allData)):
    compare = []
    compare.append(allData[i])
    distance = cdist(input,compare,metric='cosine')
    if distance < dis:
        dis = distance
        similar = i

print(similar)







