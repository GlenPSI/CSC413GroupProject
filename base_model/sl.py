import pandas as pd
import math
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from CSC413 import DATA
from trendNN_value import PricePredictionModel
import util

class SuperVisedLearning:
    def __init__(self, embedding=True, trainingName='train'):
        self.targetColName = 'Close'
        self.ticker = 'AAPL'
        self.baseDataFolder = 'historical_1min'
        self.trainStartDate = '20200101'
        self.trainEndDate = '20220301'
        self.testStartDate = '20220302'
        self.testEndDate = '20220317'
        self.trainingdone = False
        self.obWindowSize = 5
        self.dropout = 0.5
        self.saveWindow = 10000
        self.device = torch.device("cuda:0")
        self.embedding = embedding
        self.trainFilesList, self.testFilesList = util.getTrainTestDataList(self.ticker, self.baseDataFolder,
                                                                            trainStartDate=self.trainStartDate,
                                                                            trainEndDate=self.trainEndDate,
                                                                            testStartDate=self.testStartDate,
                                                                            testEndDate=self.testEndDate,
                                                                            printing=False)
        self.lenTrainFilesList = len(self.trainFilesList)
        self.epoch = 1
        self.running_loss = 0
        self.optStep = 1

        self.thisTest = f'./logs/{trainingName}/'
        if not os.path.isdir(self.thisTest):
                    os.mkdir(self.thisTest)

    def train(self):
        initDf = pd.read_csv(self.trainFilesList[random.randint(0, self.lenTrainFilesList - 1)])
        signalFeatures = initDf.to_numpy()
        nSignal = signalFeatures.shape[1]

        lstmNN = PricePredictionModel(nSignal, 1, self.device, 0, embedding=self.embedding).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstmNN.parameters(), lr = 0.001, betas=[0.9, 0.999], weight_decay=0.001)

        _loss = []
        _accuracy = []
        # Training loop
        while self.trainingdone == False:
            # load training data
            randomIndex = random.randint(0, self.lenTrainFilesList - 1)
            fileName = self.trainFilesList[randomIndex]
            
            df = pd.read_csv(self.trainFilesList[random.randint(0, self.lenTrainFilesList - 1)])
            lenDf = len(df.index)
            signalFeatures = df.to_numpy()

            target = savgol_filter(df[self.targetColName], 21, 3) 
            realValues = df[self.targetColName].to_numpy()

            # # calculate turning points
            # labels = df['vortex_diff_savgol'].to_numpy()

            # loop through the day of data
            for currentTick in range(self.obWindowSize - 1, lenDf):
                label = target[currentTick] #- realValues[currentTick]
                label = torch.FloatTensor([label]).to(self.device)
                # label = torch.squeeze(label)
                # generate state
                windowSignalFeatures = signalFeatures[(currentTick - self.obWindowSize + 1):currentTick + 1]

                # output from nn
                classOutput = lstmNN.selectClass(windowSignalFeatures)

                # optimize nn
        
                # zero the parameter gradients
                optimizer.zero_grad()
                # print("*" * 10)
                # print(classOutput)
                # print(torch.tensor(label, dtype=torch.float32).to(self.device))
                # forward + backward + optimize

                loss = criterion(classOutput, label)
                loss.backward()
                optimizer.step()

                # print statistics
                self.running_loss += loss.item()
                self.optStep += 1
                # if self.optStep % 2000 == 1999:    # print every 2000 mini-batches
                if self.optStep % self.saveWindow == 0:
                    print(str(self.epoch) + ' epoch. Loss is ' + str(self.running_loss / self.saveWindow))
                    _loss.append(self.running_loss / self.saveWindow)
                    self.running_loss = 0.0

                if self.optStep % self.saveWindow == 0:
                    currentModelPath = self.thisTest + '/sl-policy-' + str(int(self.optStep / self.saveWindow - 1)) + '.para'    
                    torch.save(lstmNN.state_dict(), currentModelPath)
                    _accuracy.append(self.test(currentModelPath, saveDF = True))
                    # print('Saved model at ' + currentModelPath)

            self.epoch += 1
            if self.epoch == 10000:
                self.trainingdone = True
                print('Training Done')
            

        return _loss, _accuracy

    def test(self, testPolicyDir, saveDF = False):
        initDf = pd.read_csv(self.trainFilesList[random.randint(0, self.lenTrainFilesList - 1)])
        signalFeatures = initDf.to_numpy()
        nSignal = signalFeatures.shape[1]
        
        initStateDict = torch.load(testPolicyDir)
        lstmNN = PricePredictionModel(nSignal, 1, self.device, 0,  embedding=self.embedding).to(self.device)
        lstmNN.load_state_dict(initStateDict)
        softmax = nn.Softmax(dim=-1)

        # policy metrics
        testDiff = 0
        testPoints = 0

        for i in range(len(self.testFilesList)):
            # load test file
            fileName = self.testFilesList[i]
            df, extraDfDataPack = pd.read_csv(self.trainFilesList[random.randint(0, self.lenTrainFilesList - 1)])
            lenDf = len(df.index)
            signalFeatures = df.loc[:, 'Close_Prev_Close_Percentage':].to_numpy()

            target = savgol_filter(df[self.targetColName], 21, 3)
            realValues = df[self.targetColName].to_numpy()

            # epoch metrics
            dayDiff = 0


            if saveDF:
                df['preditction'] = 0
                df['p-target'] = target
                df['prediction-diff'] = 0

            # loop through the day of data
            for currentTick in range(self.obWindowSize - 1, lenDf):
                label = target[currentTick] #- realValues[currentTick]

                # generate state
                windowSignalFeatures = signalFeatures[(currentTick - self.obWindowSize + 1):currentTick + 1]

                # output from nn
                classOutput = lstmNN.selectClass(windowSignalFeatures)
                out = classOutput.item()

                # if out == label:
                #     rightPoints += 1
                diff = abs(out - label)
                dayDiff += diff

                # print(str(label) + '-' + str(out))

                if saveDF:
                    df.loc[currentTick, 'preditction'] = out #+ realValues[currentTick]
                    df.loc[currentTick, 'prediction-diff'] = diff

            
            accuracy = dayDiff/lenDf
            # util.printOkBlue(fileName + ' acc: ' + str(accuracy))

            testDiff += dayDiff
            testPoints += lenDf

            if saveDF:
                currentModelPath = self.thisTest + '/' + testPolicyDir.split("/")[-1].split(".")[0] +'/'    
                if not os.path.isdir(currentModelPath):
                    os.mkdir(currentModelPath)
                df.to_csv(currentModelPath + self.testFilesList[i], index=False)

        testAccuracy = testDiff/testPoints
        util.printOkCYAN('Test acc: ' + str(testAccuracy))
        return testAccuracy

def main():
    sl = SuperVisedLearning()
    sl.train()
    testPolicyDir = ''
    # sl.test(testPolicyDir, saveDF=True)
    
if __name__ == '__main__':
    main()