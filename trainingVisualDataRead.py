###############################################
#### Written By: SATYAKI DE                ####
#### Written On: 17-Jan-2022               ####
#### Modified On 17-Jan-2022               ####
####                                       ####
#### Objective: This is the main calling   ####
#### python script that will invoke the    ####
#### clsAlhpabetReading class to initiate  ####
#### teach & perfect the model to read     ####
#### visual alphabets using Convolutional  ####
#### Neural Network (CNN).                 ####
###############################################

# We keep the setup code in a different class as shown below.
import clsAlphabetReading as ar
from clsConfig import clsConfig as cf

import datetime
import logging

###############################################
###           Global Section                ###
###############################################
# Instantiating all the three classes

x1 = ar.clsAlphabetReading()

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

        print('Started Transformation!')

        # Execute all the pass
        r1 = x1.trainModel(debugInd, var)

        if (r1 == 0):
            print('Successfully Visual Alphabet Training Completed!')
        else:
            print('Failed to complete the Visual Alphabet Training!')

        var2 = datetime.datetime.now()

        c = var2 - var1
        minutes = c.total_seconds() / 60
        print('Total difference in minutes: ', str(minutes))

        print('End Time: ', str(var1))

    except Exception as e:
        x = str(e)
        print('Error: ', x)

if __name__ == "__main__":
    main()
