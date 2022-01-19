################################################
#### Written By: SATYAKI DE                 ####
#### Written On:  15-May-2020               ####
#### Modified On: 28-Dec-2021               ####
####                                        ####
#### Objective: This script is a config     ####
#### file, contains all the keys for        ####
#### Machine-Learning & streaming dashboard.####
####                                        ####
################################################

import os
import platform as pl

class clsConfig(object):
    Curr_Path = os.path.dirname(os.path.realpath(__file__))

    os_det = pl.system()
    if os_det == "Windows":
        sep = '\\'
    else:
        sep = '/'

    conf = {
        'APP_ID': 1,
        'ARCH_DIR': Curr_Path + sep + 'arch' + sep,
        'PROFILE_PATH': Curr_Path + sep + 'profile' + sep,
        'LOG_PATH': Curr_Path + sep + 'log' + sep,
        'REPORT_PATH': Curr_Path + sep + 'report',
        'FILE_NAME': Curr_Path + sep + 'Data' + sep + 'A_Z_Handwritten_Data.csv',
        'SRC_PATH': Curr_Path + sep + 'data' + sep,
        'APP_DESC_1': 'Old Video Enhancement!',
        'DEBUG_IND': 'N',
        'INIT_PATH': Curr_Path,
        'SUBDIR': 'data',
        'SEP': sep,
        'testRatio':0.2,
        'valRatio':0.2,
        'epochsVal':8,
        'activationType':'relu',
        'activationType2':'softmax',
        'numOfClasses':26,
        'kernelSize':(3, 3),
        'poolSize':(2, 2),
        'filterVal1':32,
        'filterVal2':64,
        'filterVal3':128,
        'stridesVal':2,
        'monitorVal':'val_loss',
        'paddingVal1':'same',
        'paddingVal2':'valid',
        'reshapeVal':28,
        'reshapeVal1':(28,28),
        'patienceVal1':1,
        'patienceVal2':2,
        'sleepTime':3,
        'sleepTime1':6,
        'factorVal':0.2,
        'learningRateVal':0.001,
        'minDeltaVal':0,
        'minLrVal':0.0001,
        'verboseFlag':0,
        'modeInd':'auto',
        'shuffleVal':100,
        'DenkseVal1':26,
        'DenkseVal2':64,
        'DenkseVal3':128,
        'predParam':9,
        'word_dict':{0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'},
        'width':640,
        'height':480,
        'imgSize': (32,32),
        'threshold': 0.45,
        'imgDimension': (400, 440),
        'imgSmallDim': (7, 7),
        'imgMidDim': (28, 28),
        'reshapeParam1':1,
        'reshapeParam2':28,
        'colorFeed':(0,0,130),
        'colorPredict':(0,25,255)
    }
