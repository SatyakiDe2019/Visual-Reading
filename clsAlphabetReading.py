###############################################
#### Written By: SATYAKI DE                ####
#### Written On: 17-Jan-2022               ####
#### Modified On 17-Jan-2022               ####
####                                       ####
#### Objective: This python script will    ####
#### teach & perfect the model to read     ####
#### visual alphabets using Convolutional  ####
#### Neural Network (CNN).                 ####
###############################################

from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
import pandas as p
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle

import pickle

import os
import platform as pl

from clsConfig import clsConfig as cf

class clsAlphabetReading:
    def __init__(self):
        self.sep = str(cf.conf['SEP'])
        self.Curr_Path = str(cf.conf['INIT_PATH'])
        self.fileName = str(cf.conf['FILE_NAME'])
        self.testRatio = float(cf.conf['testRatio'])
        self.valRatio = float(cf.conf['valRatio'])
        self.epochsVal = int(cf.conf['epochsVal'])
        self.activationType = str(cf.conf['activationType'])
        self.activationType2 = str(cf.conf['activationType2'])
        self.numOfClasses = int(cf.conf['numOfClasses'])
        self.kernelSize = cf.conf['kernelSize']
        self.poolSize = cf.conf['poolSize']
        self.filterVal1 = int(cf.conf['filterVal1'])
        self.filterVal2 = int(cf.conf['filterVal2'])
        self.filterVal3 = int(cf.conf['filterVal3'])
        self.stridesVal = int(cf.conf['stridesVal'])
        self.monitorVal = str(cf.conf['monitorVal'])
        self.paddingVal1 = str(cf.conf['paddingVal1'])
        self.paddingVal2 = str(cf.conf['paddingVal2'])
        self.reshapeVal = int(cf.conf['reshapeVal'])
        self.reshapeVal1 = cf.conf['reshapeVal1']
        self.patienceVal1 = int(cf.conf['patienceVal1'])
        self.patienceVal2 = int(cf.conf['patienceVal2'])
        self.sleepTime = int(cf.conf['sleepTime'])
        self.sleepTime1 = int(cf.conf['sleepTime1'])
        self.factorVal = float(cf.conf['factorVal'])
        self.learningRateVal = float(cf.conf['learningRateVal'])
        self.minDeltaVal = int(cf.conf['minDeltaVal'])
        self.minLrVal = float(cf.conf['minLrVal'])
        self.verboseFlag = int(cf.conf['verboseFlag'])
        self.modeInd = str(cf.conf['modeInd'])
        self.shuffleVal = int(cf.conf['shuffleVal'])
        self.DenkseVal1 = int(cf.conf['DenkseVal1'])
        self.DenkseVal2 = int(cf.conf['DenkseVal2'])
        self.DenkseVal3 = int(cf.conf['DenkseVal3'])
        self.predParam = int(cf.conf['predParam'])
        self.word_dict = cf.conf['word_dict']

    def applyCNN(self, X_Train, Y_Train_Catg, X_Validation, Y_Validation_Catg):
        try:
            testRatio = self.testRatio
            epochsVal = self.epochsVal
            activationType = self.activationType
            activationType2 = self.activationType2
            numOfClasses = self.numOfClasses
            kernelSize = self.kernelSize
            poolSize = self.poolSize
            filterVal1 = self.filterVal1
            filterVal2 = self.filterVal2
            filterVal3 = self.filterVal3
            stridesVal = self.stridesVal
            monitorVal = self.monitorVal
            paddingVal1 = self.paddingVal1
            paddingVal2 = self.paddingVal2
            reshapeVal = self.reshapeVal
            patienceVal1 = self.patienceVal1
            patienceVal2 = self.patienceVal2
            sleepTime = self.sleepTime
            sleepTime1 = self.sleepTime1
            factorVal = self.factorVal
            learningRateVal = self.learningRateVal
            minDeltaVal = self.minDeltaVal
            minLrVal = self.minLrVal
            verboseFlag = self.verboseFlag
            modeInd = self.modeInd
            shuffleVal = self.shuffleVal
            DenkseVal1 = self.DenkseVal1
            DenkseVal2 = self.DenkseVal2
            DenkseVal3 = self.DenkseVal3

            model = Sequential()

            model.add(Conv2D(filters=filterVal1, kernel_size=kernelSize, activation=activationType, input_shape=(28,28,1)))
            model.add(MaxPool2D(pool_size=poolSize, strides=stridesVal))

            model.add(Conv2D(filters=filterVal2, kernel_size=kernelSize, activation=activationType, padding = paddingVal1))
            model.add(MaxPool2D(pool_size=poolSize, strides=stridesVal))

            model.add(Conv2D(filters=filterVal3, kernel_size=kernelSize, activation=activationType, padding = paddingVal2))
            model.add(MaxPool2D(pool_size=poolSize, strides=stridesVal))

            model.add(Flatten())

            model.add(Dense(DenkseVal2,activation = activationType))
            model.add(Dense(DenkseVal3,activation = activationType))

            model.add(Dense(DenkseVal1,activation = activationType2))

            model.compile(optimizer = Adam(learning_rate=learningRateVal), loss='categorical_crossentropy', metrics=['accuracy'])
            reduce_lr = ReduceLROnPlateau(monitor=monitorVal, factor=factorVal, patience=patienceVal1, min_lr=minLrVal)
            early_stop = EarlyStopping(monitor=monitorVal, min_delta=minDeltaVal, patience=patienceVal2, verbose=verboseFlag, mode=modeInd)


            fittedModel = model.fit(X_Train, Y_Train_Catg, epochs=epochsVal, callbacks=[reduce_lr, early_stop],  validation_data = (X_Validation,Y_Validation_Catg))

            return (model, fittedModel)

        except Exception as e:
            x = str(e)
            model = Sequential()
            print('Error: ', x)

            return (model, model)

    def trainModel(self, debugInd, var):
        try:
            sep = self.sep
            Curr_Path = self.Curr_Path
            fileName = self.fileName
            epochsVal = self.epochsVal
            valRatio = self.valRatio
            predParam = self.predParam
            testRatio = self.testRatio
            reshapeVal = self.reshapeVal
            numOfClasses = self.numOfClasses
            sleepTime = self.sleepTime
            sleepTime1 = self.sleepTime1
            shuffleVal = self.shuffleVal
            reshapeVal1 = self.reshapeVal1

            # Dictionary for getting characters from index values
            word_dict = self.word_dict

            print('File Name: ', str(fileName))

            # Read the data
            df_HW_Alphabet = p.read_csv(fileName).astype('float32')

            # Sample Data
            print('Sample Data: ')
            print(df_HW_Alphabet.head())

            # Split data the (x - Our data) & (y - the prdict label)
            x = df_HW_Alphabet.drop('0',axis = 1)
            y = df_HW_Alphabet['0']


            # Reshaping the data in csv file to display as an image
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = testRatio)
            X_Train, X_Validation, Y_Train, Y_Validation = train_test_split(X_Train, Y_Train, test_size = valRatio)

            X_Train = np.reshape(X_Train.values, (X_Train.shape[0], reshapeVal, reshapeVal))
            X_Test = np.reshape(X_Test.values, (X_Test.shape[0], reshapeVal, reshapeVal))
            X_Validation = np.reshape(X_Validation.values, (X_Validation.shape[0], reshapeVal, reshapeVal))


            print("Train Data Shape: ", X_Train.shape)
            print("Test Data Shape: ", X_Test.shape)
            print("Validation Data shape: ", X_Validation.shape)

            # Plotting the number of alphabets in the dataset
            Y_Train_Num = np.int0(y)
            count = np.zeros(numOfClasses, dtype='int')
            for i in Y_Train_Num:
                count[i] +=1

            alphabets = []
            for i in word_dict.values():
                alphabets.append(i)

            fig, ax = plt.subplots(1,1, figsize=(7,7))
            ax.barh(alphabets, count)

            plt.xlabel("Number of elements ")
            plt.ylabel("Alphabets")
            plt.grid()
            plt.show(block=False)
            plt.pause(sleepTime)
            plt.close()

            # Shuffling the data
            shuff = shuffle(X_Train[:shuffleVal])

            # Model reshaping the training & test dataset
            X_Train = X_Train.reshape(X_Train.shape[0],X_Train.shape[1],X_Train.shape[2],1)
            print("Shape of Train Data: ", X_Train.shape)

            X_Test = X_Test.reshape(X_Test.shape[0], X_Test.shape[1], X_Test.shape[2],1)
            print("Shape of Test Data: ", X_Test.shape)

            X_Validation = X_Validation.reshape(X_Validation.shape[0], X_Validation.shape[1], X_Validation.shape[2],1)
            print("Shape of Validation data: ", X_Validation.shape)

            # Converting the labels to categorical values
            Y_Train_Catg = to_categorical(Y_Train, num_classes = numOfClasses, dtype='int')
            print("Shape of Train Labels: ", Y_Train_Catg.shape)

            Y_Test_Catg = to_categorical(Y_Test, num_classes = numOfClasses, dtype='int')
            print("Shape of Test Labels: ", Y_Test_Catg.shape)

            Y_Validation_Catg = to_categorical(Y_Validation, num_classes = numOfClasses, dtype='int')
            print("Shape of validation labels: ", Y_Validation_Catg.shape)

            model, history = self.applyCNN(X_Train, Y_Train_Catg, X_Validation, Y_Validation_Catg)

            print('Model Summary: ')
            print(model.summary())

            # Displaying the accuracies & losses for train & validation set
            print("Validation Accuracy :", history.history['val_accuracy'])
            print("Training Accuracy :", history.history['accuracy'])
            print("Validation Loss :", history.history['val_loss'])
            print("Training Loss :", history.history['loss'])

            # Displaying the Loss Graph
            plt.figure(1)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.legend(['training','validation'])
            plt.title('Loss')
            plt.xlabel('epoch')
            plt.show(block=False)
            plt.pause(sleepTime1)
            plt.close()

            # Dsiplaying the Accuracy Graph
            plt.figure(2)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.legend(['training','validation'])
            plt.title('Accuracy')
            plt.xlabel('epoch')
            plt.show(block=False)
            plt.pause(sleepTime1)
            plt.close()

            # Making the model to predict
            pred = model.predict(X_Test[:predParam])

            print('Test Details::')
            print('X_Test: ', X_Test.shape)
            print('Y_Test_Catg: ', Y_Test_Catg.shape)

            try:
                score = model.evaluate(X_Test, Y_Test_Catg, verbose=0)
                print('Test Score = ', score[0])
                print('Test Accuracy = ', score[1])
            except Exception as e:
                x = str(e)
                print('Error: ', x)

            # Displaying some of the test images & their predicted labels
            fig, ax = plt.subplots(3,3, figsize=(8,9))
            axes = ax.flatten()

            for i in range(9):
                axes[i].imshow(np.reshape(X_Test[i], reshapeVal1), cmap="Greys")
                pred = word_dict[np.argmax(Y_Test_Catg[i])]
                print('Prediction: ', pred)
                axes[i].set_title("Test Prediction: " + pred)
                axes[i].grid()
            plt.show(block=False)
            plt.pause(sleepTime1)
            plt.close()

            fileName = Curr_Path + sep + 'Model' + sep + 'model_trained_' + str(epochsVal) + '.p'
            print('Model Name: ', str(fileName))

            pickle_out = open(fileName, 'wb')
            pickle.dump(model, pickle_out)
            pickle_out.close()

            return 0
        except Exception as e:
            x = str(e)
            print('Error: ', x)

            return 1
