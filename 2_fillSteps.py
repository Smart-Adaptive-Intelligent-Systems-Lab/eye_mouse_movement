import csv
import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#import pafy
import csv
# import pandas as pd

from skimage.draw import line


def readFile(filename):
    lst = []
    with open(filename, mode='r') as f:
        data = csv.DictReader(f)
        for row in data:
            lst.append(dict(row))

    return lst

directions = [
    0, # stay in current position
    1, # move up
    2, # move up/left diagonal
    3, # move left
    4, # move down/left diagonal
    5, # move down
    6, # move down/right diagonal
    7, # move right
    8  # move up/right diagonal
]

object_points = {
    # (y, x)
    '5,1': 1,
    '5,2': 1,

    '8,4': 2,
    '9,4': 2,

    '9,6': 3,

    '8,6': 4,

    '7,7': 5,

    '6,8': 6,

    '4,8': 7,
    '9,10': 8,
    '5,10': 9,
    '9,11': 10,
    '5,11': 11,
    '5,13': 12

}

'''
If we have 9 directions,  [2,8]=>[4,6] will be  [2,8, action number] => [3,7, action number] => [4,6, action]

Possible OOO:
    1) calculate line from point1 to point2
        a) evaluate every point along line calculated
        b) determine direction of each point evaluated
            if x1 < x2 and y1 < y2: move down/left
    2) repeat step 1 for every pair in list...

exploratory 

'''

def determineAction(point, prev):
    action = 0
    # moving up
    if (point[0] < prev[0] and point[1] == prev[1]): action = 1
        # print(str(point) + ',1')

    # moving up/left diagonal
    elif (point[0] < prev[0] and point[1] < prev[1]): action = 2
        # print(str(point) + ',2')

    # moving left
    elif (point[0] == prev[0] and point[1] < prev[1]): action = 3
        # print(str(point) + ',3')

    # moving down/left diagonal
    elif (point[0] > prev[0] and point[1] < prev[1]): action = 4
        # print(str(point) + ',4')

    # moving down
    elif (point[0] > prev[0] and point[1] == prev[1]): action = 5
        # print(str(point) + ',5')

    # moving down/right diagonal
    elif (point[0] > prev[0] and point[1] > prev[1]): action = 6
        # print(str(point) + ',6')

    # moving right
    elif (point[0] == prev[0] and point[1] > prev[1]): action = 7
        # print(str(point) + ',7')

    # moving up/right diagonal
    elif (point[0] < prev[0] and point[1] > prev[1]): action = 8
        # print(str(point) + ',8')

    return action

def computeSteps(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]

def getObject(p):
    # print('p: ', p, object_points)
    k = str(p[0]) + ',' + str(p[1])
    if k in object_points:
        return object_points[k]
    else: return 0

def equiv(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]

if __name__ == "__main__":

    filename = 'generated_bare_trajectories_c4_gaze.csv'
    data = readFile(filename)

    # printing titles
    print('point,action,object,ms')

    p1 = np.asarray([2, 8])
    for row in data:    
        # print('---> ', row)
        p2 = row['point'].split(',')
        p2 = np.asarray([int(p2[0]), int(p2[1])])

        # If no movement, action number is 0
        if equiv(p1, p2): 
            print(str(p1[1]) + ',' + str(p1[0]) + ',0,' + str(getObject(p1)) + ',' + str(row[' ms']))
            continue

        d1, d2 = computeSteps(p1, p2)
        if d1 < 4 and d2 < 4:
            x1 = int(p1[0])
            y1 = int(p1[1])

            x2 = int(p2[0])
            y2 = int(p2[1])

            # Generate line from p1 to p2
            discrete_line = list(zip(*line(x1, y1, x2, y2)))

            prev = (x1, y1)
            for point in discrete_line:
                
                action = determineAction(point, prev)
                currentobj = getObject(point)

                print(str(point[1]) + ',' + str(point[0]) + ',' + str(action) + ',' + str(currentobj) + ',' + str(row[' ms']))
                prev = point
                # print('\n')

        if d1 > 3 or d2 > 3:
            print('Process {}->{}'.format(p1, p2))

        p1 = p2
        

            
