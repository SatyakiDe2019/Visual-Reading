###############################################
#### Written By: SATYAKI DE                ####
#### Written On: 18-Jan-2022               ####
#### Modified On 18-Jan-2022               ####
####                                       ####
#### Objective: This python script will    ####
#### scan the live video feed from the     ####
#### web-cam & predict the alphabet that   ####
#### read it.                              ####
###############################################

# We keep the setup code in a different class as shown below.
from clsConfig import clsConfig as cf

import datetime
import logging
import cv2
import pickle
import numpy as np
###############################################
###           Global Section                ###
###############################################

sep = str(cf.conf['SEP'])
Curr_Path = str(cf.conf['INIT_PATH'])
fileName = str(cf.conf['FILE_NAME'])
epochsVal = int(cf.conf['epochsVal'])
numOfClasses = int(cf.conf['numOfClasses'])
word_dict = cf.conf['word_dict']
width = int(cf.conf['width'])
height = int(cf.conf['height'])
imgSize = cf.conf['imgSize']
threshold = float(cf.conf['threshold'])
imgDimension = cf.conf['imgDimension']
imgSmallDim = cf.conf['imgSmallDim']
imgMidDim = cf.conf['imgMidDim']
reshapeParam1 = int(cf.conf['reshapeParam1'])
reshapeParam2 = int(cf.conf['reshapeParam2'])
colorFeed = cf.conf['colorFeed']
colorPredict = cf.conf['colorPredict']
###############################################
###    End of Global Section                ###
###############################################

def main():
    try:
        # Other useful variables
        debugInd = 'Y'
        var = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        var1 = datetime.datetime.now()

        print('Start Time: ', str(var))
        # End of useful variables

        # Initiating Log Class
        general_log_path = str(cf.conf['LOG_PATH'])

        # Enabling Logging Info
        logging.basicConfig(filename=general_log_path + 'restoreVideo.log', level=logging.INFO)

        print('Started Live Streaming!')

        cap = cv2.VideoCapture(0)
        cap.set(3, width)
        cap.set(4, height)

        fileName = Curr_Path + sep + 'Model' + sep + 'model_trained_' + str(epochsVal) + '.p'
        print('Model Name: ', str(fileName))

        pickle_in = open(fileName, 'rb')
        model = pickle.load(pickle_in)

        while True:
            status, img = cap.read()

            if status == False:
                break

            img_copy = img.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, imgDimension)

            img_copy = cv2.GaussianBlur(img_copy, imgSmallDim, 0)
            img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            bin, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

            img_final = cv2.resize(img_thresh, imgMidDim)
            img_final = np.reshape(img_final, (reshapeParam1,reshapeParam2,reshapeParam2,reshapeParam1))


            img_pred = word_dict[np.argmax(model.predict(img_final))]

            # Extracting Probability Values
            Predict_X = model.predict(img_final)
            probVal = round(np.amax(Predict_X) * 100)

            cv2.putText(img, "Live Feed : (" + str(probVal) + "%) ", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = colorFeed)
            cv2.putText(img, "Prediction: " + img_pred, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = colorPredict)

            cv2.imshow("Original Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                r1=0
                break

        if (r1 == 0):
            print('Successfully Alphabets predicted!')
        else:
            print('Failed to predict alphabet!')

        var2 = datetime.datetime.now()

        c = var2 - var1
        minutes = c.total_seconds() / 60
        print('Total Run Time in minutes: ', str(minutes))

        print('End Time: ', str(var1))

    except Exception as e:
        x = str(e)
        print('Error: ', x)

if __name__ == "__main__":
    main()
