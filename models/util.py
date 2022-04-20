
import json
import pandas as pd
import os 
import datetime
from CSC413 import DATA

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # sample usage : print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
def printOkGreen(msg):
    print(f'{bcolors.OKGREEN}{msg}{bcolors.ENDC}')
def printWarning(msg):
    print(f'{bcolors.WARNING}{msg}{bcolors.ENDC}')
def printOkBlue(msg):
    print(f'{bcolors.OKBLUE}{msg}{bcolors.ENDC}')
def printOkCYAN(msg):
    print(f'{bcolors.OKCYAN}{msg}{bcolors.ENDC}')
def printFail(msg):
    print(f'{bcolors.FAIL}{msg}{bcolors.ENDC}')
    

def cls(line): print("\n" * 30)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def saveDataFrame(fileName, dataframe, directory, i=False, js = False,csv = True):
    
    if csv:    
        pd.DataFrame.from_dict(
            dataframe
            ,orient='columns'
        ).to_csv(os.path.join(directory, f'{fileName}.csv'),index=i)
    if js:    
        with open(os.path.join(directory, f'{fileName}.json'), 'w', encoding='utf-8') as f:
            json.dump(dataframe, f, ensure_ascii=False, indent=4)

######################## Data Prep ##############################


stocksAttributes = [{'name':'AAPL',
                      'trainStart':'20200101',
                      'trainEnd':'20220306',
                      'testStart':'20220307',
                      'testEnd':'20220318'
                     },
                     {'name':'QQQ',
                      'trainStart':'20200101',
                      'trainEnd':'20220306',
                      'testStart':'20220307',
                      'testEnd':'20220318'
                     },
                     {'name':'TSLA',
                      'trainStart':'20200101',
                      'trainEnd':'20220306',
                      'testStart':'20220307',
                      'testEnd':'20220318'
                     },]

def getStockDir(stockTicker,folderName, dataFolder=None):
    if dataFolder==None:
        dataFolder=DATA
    return dataFolder + '/'+ stockTicker+'/'+folderName+'/'


def getDataList(stockTicker,baseDataFolder, dataFolder=None): 
    dir = getStockDir(stockTicker,baseDataFolder, dataFolder)
    allFiles=list(filter( lambda a: 'csv' in a ,os.listdir(dir))) # load all days of data in transformed data folder, filters non csv files
    trainDirExist=os.path.exists(dir + '.DS_Store')
    if trainDirExist:
        allFiles.remove('.DS_Store')
    allFiles.sort()
    
    return allFiles


def getAllStockList():
    index = 0
    stockLists =[]
    while(index<len(stocksAttributes)):
        stockLists.append(stocksAttributes[index].get('name'))
        index = index + 1 
    return stockLists


def isStock(stockTicker):
    stockLists = getAllStockList()
    index = 0
    while(index<len(stockLists)):
        if stockLists[index] == stockTicker:
            return True
        index = index + 1
    return False
            

def getDateTime(dateString):
    year = dateString[0:4]
    month = dateString[4:6]
    day = dateString[6:8]
    return datetime.datetime(int(year), int(month), int(day))

def getTrainTestDataList(stockTicker,baseDataFolder, trainStartDate,trainEndDate,testStartDate,testEndDate, printing=True):    

    if not isStock(stockTicker):
        error = "The given stock name " + stockTicker + " is not in the training data stock list"
        print(error)
        return
    
    allFiles=getDataList(stockTicker,baseDataFolder)
    assert len(allFiles) > 0

    trainList = []
    testList = []
    # trainStartDate,trainEndDate,testStartDate,testEndDate = getFourDates(stockTicker)
    index = 0


    while index<len(allFiles):
        curDate = getDateTime(allFiles[index].split('-')[1].split('.')[0])

        if curDate >= getDateTime(trainStartDate) and curDate<=getDateTime(trainEndDate): 
            trainList.append(allFiles[index])

        if curDate >= getDateTime(testStartDate) and curDate<=getDateTime(testEndDate): 
            testList.append(allFiles[index])            

        index = index + 1
    
    trainDFList, testDFlist = [], []
    
    dataDir = getStockDir(stockTicker,baseDataFolder)
    for file in trainList:
        trainDFList.append(pd.read_csv(f'{dataDir}/{file}'))
    
    for file in testList:  
        testDFlist.append(pd.read_csv(f'{dataDir}/{file}'))
    
    if printing:
        print('trainList: \n {} \n'.format(trainList))
        print('testList: \n {} \n'.format(testList))

    return trainList,testList, trainDFList, testDFlist
