import pandas as pd
import numpy as np
import scipy
from random import shuffle
import random
import matplotlib.pyplot as plt

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
    data[' GT_W']  = globalSpeedPostProcessingGet(data[' GT_THETA'],data[' TIMESTAMP'])

    # inputs will be only m1, m2, m3  and m4, theta
    inputs = []
    for index in range(len(data[' GT_W'])):
        inputs.append(np.array([data[' ROBOT_M1'][index], data[' ROBOT_M2'][index], data[' ROBOT_M3'][index], data[' ROBOT_M4'][index], data[' GT_THETA'][index]]))
    # outputs will gt_vx, gt_vy, and gt_w
    gt_outputs = []
    for index in range(len(data[' GT_W'])):
        gt_outputs.append(np.array([data[' GT_VX'][index], data[' GT_VY'][index], data[' GT_W'][index]]))

    return data, inputs, gt_outputs

#_, x0, y0 = dataGet("data/quadrado_opt_1_1.csv")
#_, x1, y1 = dataGet("data/quadrado_opt_1_2.csv")
#_, x2, y2 = dataGet("data/quadrado_opt_1_3.csv")
#_, x3, y3 = dataGet("data/quadrado_opt_1_4.csv")
#_, x4, y4 = dataGet("data/quadrado_opt_1_5.csv")
#_, x5, y5 = dataGet("data/quadrado_opt_2_1.csv")
#_, x6, y6 = dataGet("data/quadrado_opt_2_2.csv")
#_, x7, y7 = dataGet("data/quadrado_opt_2_3.csv")
#_, x8, y8 = dataGet("data/quadrado_opt_2_4.csv")
#_, x9, y9 = dataGet("data/quadrado_opt_2_5.csv")


#x0 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
#y0 = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9

'''
now that i have complete set of inputs and ground truth outputs, i want to break them down into arrays of size N to be fed to the training set
'''
def createTestTrainSets(x, y, N, trainRatio = 0.6, valRatio = 0.2, testRatio = 0.2):
    trainSize = int(trainRatio*int(len(x)/N))
    valSize   = int(valRatio*int(len(x)/N))
    testSize  = int(testRatio*int(len(x)/N))

    x_list = []
    y_list = []
    #print(int(len(x)/N))
    #print(trainSize, valSize, testSize)
    for iter in range(int(len(x)/N)):
        x_list.append(x[iter*N:(iter+1)*N])
        y_list.append(y[iter*N:(iter+1)*N])
    shuffle(x_list)
    shuffle(y_list)

    trainTuple = []
    valTuple = []
    testTuple = []
    for i in range(trainSize):
        trainTuple.append((x_list[i], y_list[i]))
    for i in range(valSize):
        valTuple.append((x_list[i+trainSize], y_list[i+trainSize]))
    for i in range(testSize):
        testTuple.append((x_list[i+trainSize+valSize], y_list[i+trainSize+valSize]))
    return trainTuple, valTuple, testTuple

#trainTuple, valTuple, testTuple = createTestTrainSets(x0, y0, 10)

def dataPlotMulti(dataXList, dataYlist, plotTitle):
    # Plotting the data
    plt.figure(figsize=(10, 6))
    for iter in range(len(dataXList)):
        dataY = dataYlist[iter]
        dataX = dataXList[iter]
        plt.plot(dataX, dataY, label=dataY.name)#, marker='o', linestyle='-')
    plt.title(plotTitle)
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")

    plt.grid(True)
    plt.show()

def sequenceChunksBreak(inputs, targets, chunkLength=100):
    ''' breaks the two arrays inputs and targets into chunks of length chunkLength'''
    if (len(inputs)!=len(targets)):
        print("error! diff size lists")
    totalSize = len(inputs)
    numOfChunks = int(totalSize/chunkLength)
    inputChunks = []
    targetChunks = []
    for iter in range(numOfChunks):
        #print(print("inputs[iter:iter+chunkLength] ",iter,  inputs[iter:iter+chunkLength]))
        inputChunks.append(inputs[iter*chunkLength:(iter+1)*chunkLength])
        targetChunks.append(targets[iter*chunkLength:(iter+1)*chunkLength])
    return inputChunks, targetChunks, inputs, targets


def dataGet_v2(paths, sequenceLength, trainRatio=0.8, valRatio=0.1, testRatio=0.1):
    inputs = []
    targets = []
    sequences = []
    goal_size = 0
    test_ins = []
    test_outs = []

    for path in paths:
        data, x0, y0 = dataGet(path)

        # break here into chunks of sequenceLength
        inputChunks, targetChunks, x, y = sequenceChunksBreak(x0,y0,chunkLength=sequenceLength)
        inputs = inputs + inputChunks
        targets = targets + targetChunks
        test_ins += x
        test_outs += y
        goal_size += int(len(x0)/sequenceLength)


    inputs = random.sample(inputs, len(inputs))
    targets = random.sample(targets, len(targets))

    trainSize = int(trainRatio*int(len(inputs)))
    valSize   = int(valRatio*int(len(inputs)))
    testSize  = int(testRatio*int(len(inputs)))

    train_input = np.array(inputs[0:trainSize])
    train_target = np.array(targets[0:trainSize])

    cv_input = np.array(inputs[trainSize:trainSize+valSize])
    cv_target = np.array(targets[trainSize:trainSize+valSize])

    test_input = np.array(inputs[trainSize+valSize:])
    test_target = np.array(targets[trainSize+valSize:])

    return train_input, train_target, data, cv_input, cv_target