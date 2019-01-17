import math
import os
import sys
sys.path.append('../')
from task1 import *
from task2 import *
from task3 import *
from task3mask import *
from task4 import *


#FILE='eg.txt'


def classify(FILE):
    try:
        task1file = '../media/task1file.jpg'    # croped head
        task2file = '../media/task2file.jpg'    # similiar cartoon
        task4file = '../media/task4file.jpg'    # pre-processed cartoon
        task3mask = '../media/task3mask.jpg'    # masked cartoon
        task3file = '../media/op3.jpg'    # finalized cartoon
        task1(FILE, task1file)
        task2(task1file, task2file)
        task4(task2file, task1file)
        os.system('cp ../task4/output/399.png ../media/task4file.jpg')
        task3mask(task2file, task3mask)
        task3main(task4file, task3mask, '../modelsfile/rain/conditional_style_network.ckpt', task3file)
        os.system('rm ../media/task*')
    except:
        pass
    return(0)


