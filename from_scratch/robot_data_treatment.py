import pandas as pd
import numpy as np
import scipy
from random import shuffle

def globalSpeedPostProcessingGet(posList, timestampList):
    speeds = []
    numberFilterCoefs = 20
    for i in range(len(posList)):
        if i == 0:
            speeds.append(0)
        else:
            derivative = (posList[i] - posList[i-1])/(timestampList[i] - timestampList[i-1])*1000.0 #timestamp unit is ms
            speeds.append(derivative)

    speeds = scipy.signal.lfilter([1.0/numberFilterCoefs]*numberFilterCoefs, 1, speeds)
    return speeds

def dataGet(path):
    # Read data from CSV file
    data = pd.read_csv(path)

    data[' GT_X']= data[' VISION_X'] - data[' VISION_X'][0]
    data[' GT_Y']= data[' VISION_Y'] - data[' VISION_Y'][0]
    data[' GT_THETA'] = data[' VISION_W']
    data[' GT_THETA'] = np.unwrap(data[' GT_THETA'], period=np.pi)

    data[' GT_VX'] = globalSpeedPostProcessingGet(data[' GT_X'],data[' TIMESTAMP'])
    data[' GT_VY'] = globalSpeedPostProcessingGet(data[' GT_Y'],data[' TIMESTAMP'])
    data[' GT_W'] = globalSpeedPostProcessingGet(data[' GT_THETA'],data[' TIMESTAMP'])

    # inputs will be only m1, m2, m3  and m4
    inputs = []
    for index in range(len(data[' GT_W'])):
        inputs.append(np.array([data[' ROBOT_M1'][index], data[' ROBOT_M2'][index], data[' ROBOT_M3'][index], data[' ROBOT_M4'][index]]))
    # outputs will gt_vx, gt_vy, and gt_w
    gt_outputs = []
    for index in range(len(data[' GT_W'])):
        gt_outputs.append(np.array([data[' GT_VX'][index], data[' GT_VY'][index], data[' GT_W'][index]]))

    return data, inputs, gt_outputs

_, x0, y0 = dataGet("data/quadrado_opt_1_1.csv")
_, x1, y1 = dataGet("data/quadrado_opt_1_2.csv")
_, x2, y2 = dataGet("data/quadrado_opt_1_3.csv")
_, x3, y3 = dataGet("data/quadrado_opt_1_4.csv")
_, x4, y4 = dataGet("data/quadrado_opt_1_5.csv")
_, x5, y5 = dataGet("data/quadrado_opt_2_1.csv")
_, x6, y6 = dataGet("data/quadrado_opt_2_2.csv")
_, x7, y7 = dataGet("data/quadrado_opt_2_3.csv")
_, x8, y8 = dataGet("data/quadrado_opt_2_4.csv")
_, x9, y9 = dataGet("data/quadrado_opt_2_5.csv")

print(len(x0))
x0 = x0 #+ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
y0 = y0 #+ y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9
print(len(x0))

'''
now that i have complete set of inputs and ground truth outputs, i want to break them down into arrays of size N to be fed to the training set
'''
def createTestTrainSets(x, y, N, trainRatio = 0.6, valRatio = 0.2, testRatio = 0.2):
    trainSize = int(trainRatio*int(len(x)/N))
    valSize   = int(valRatio*int(len(x)/N))
    testSize  = int(testRatio*int(len(x)/N))

    x_list = []
    y_list = []
    print(int(len(x)/N))
    print(trainSize, valSize, testSize)
    for iter in range(int(len(x)/N)):
        x_list.append(x[iter:iter+N])
        y_list.append(y[iter:iter+N])
    shuffle(x_list)
    shuffle(y_list)




createTestTrainSets(x0, y0, 10)