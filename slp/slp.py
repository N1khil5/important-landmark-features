import random
import csv
import math
import matplotlib.pyplot as plt
#Importing the training dataset from the .csv file
with open('Train.csv', newline='') as csvfile:
    aReader = list(csv.reader(csvfile, delimiter=','))

#Initialising the weights and the learning rate
MaxWeight = 0.3
v = [random.uniform(-MaxWeight, MaxWeight) for x in range(7)]
alpha = 0.3
mses = []

#Running this loop 1000 time. Each time is 1 epoch.
for j in range(0, 1000):

#Range 1 to 37 is the patterns in the dataset
    for i in range(1, 37):

#Each value of x[0] to x[5] corresponds to an input feature of the landmark.
#Value of t is the annual popularity feature of the landmark
        x0 = float(aReader[i][1])
        x1 = float(aReader[i][2])
        x2 = float(aReader[i][3])
        x3 = float(aReader[i][4])
        x4 = float(aReader[i][5])
        x5 = float(aReader[i][6])
        t = float(aReader[i][7])

#Sum of Inputs
        ySum = v[0] * x0 + v[1] * x1 + v[2] * x2 + v[3] * x3 + v[4] * x4 + v[5] * x5 + v[6]
#Activation on output
        y = (2.0 / (1.0 + math.exp(-ySum))) - 1.0

#Finding the error correction term
        delta = (t - y) * (0.5 * (1.0 + y) * (1.0 - y))

#Computing the weight correction term for each input
        wCorrection0 = alpha * delta * x0
        wCorrection1 = alpha * delta * x1
        wCorrection2 = alpha * delta * x2
        wCorrection3 = alpha * delta * x3
        wCorrection4 = alpha * delta * x4
        wCorrection5 = alpha * delta * x5

#Computing the bias correction term
        biasCorrection = alpha * delta

#Updating the weights and bias
        v[0] = v[0] + wCorrection0
        v[1] = v[1] + wCorrection1
        v[2] = v[2] + wCorrection2
        v[3] = v[3] + wCorrection3
        v[4] = v[4] + wCorrection4
        v[5] = v[5] + wCorrection5
        v[6] = v[6] + biasCorrection

#Initialising sum error
    SumError = 0
    for i in range(1, 37):
        x0 = float(aReader[i][1])
        x1 = float(aReader[i][2])
        x2 = float(aReader[i][3])
        x3 = float(aReader[i][4])
        x4 = float(aReader[i][5])
        x5 = float(aReader[i][6])
        t = float(aReader[i][7])
        ySum = v[0] * x0 + v[1] * x1 + v[2] * x2 + v[3] * x3 + v[4] * x4 + v[5] * x5 + v[6]
        y = 2.0/(1.0 + math.exp(-ySum)) - 1.0
        SumError += (t - y) ** 2
        print('target', t, 'actual', y)

#Calculating mean squared error of the training set
    MSE = SumError / 36.0
    print("MSE = ", MSE)
    mses.append(MSE)
plt.plot(mses)
plt.show()

#Importing the test dataset from the .csv file
with open('Test.csv', newline='') as csvfile:
    testReader = list(csv.reader(csvfile, delimiter=','))

SumError = 0
for l in range(1, 11):
    t0 = float(testReader[l][1])
    t1 = float(testReader[l][2])
    t2 = float(testReader[l][3])
    t3 = float(testReader[l][4])
    t4 = float(testReader[l][5])
    t5 = float(testReader[l][6])
    t = float(testReader[l][7])
    ySum = v[0] * t0 + v[1] * t1 + v[2] * t2 + v[3] * t3 + v[4] * t4 + v[5] * t5 + v[6]
    y = (2.0 / (1.0 + math.exp(-ySum))) - 1.0
    SumError += (t - y) ** 2
    print('test target', t, 'actual', y)

#Calculating the value of the test mean squared error
testMSE = SumError / 10.0
print('testMSE = ', testMSE)

#printing the updated weights for each input feature
print('v[0]',v[0])
print('v[1]',v[1])
print('v[2]',v[2])
print('v[3]',v[3])
print('v[4]',v[4])
print('v[5]',v[5])

