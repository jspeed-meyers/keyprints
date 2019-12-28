##########----------##########----------##########----------##########----------

## KEYPRINTS (KEY STROKES AS FINGERPRINTS)
## JOHN SPEED MEYERS
## 2019/24/15

## DESCRIPTION: Build program to test if keystroke dynamics can serve as a 
## reliable identifier between two people. This versions takes input
## from two users, converts the keystroke input to features, trains a 
## model to recognize keystrokes from two users, and make predictions on
## which user is typing on future keystrokes.

##########----------##########----------##########----------##########----------

###########################
## Import necessary modules
###########################

## Python standard libraries
import time
import string
import math
import logging
import os
import argparse
import sys

## Suppress pygame message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

## External libraries
import pygame ## key module for keystroke measurement
import numpy as np
import pandas as pd
from datetime import datetime

## Add parser folder to path
sys.path.append("parser")
sys.path.append("modeling")
sys.path.append("interface")

## Import project-specfic custom functions
from parser_util import analyze
from modeling import train
from modeling import predict
from interface import collectData

## Main function calls to collect keyprints, create model distinguishing
## keyprints of user 1 from user 2, and then predict whether future keyprints
## are from user 1 or user 2
if __name__ == '__main__':
    
    ## Configure logging for keyprints module
    logging.basicConfig(filename='keyprints.log',
                        level=logging.INFO,
                        filemode='w')

    logging.info("Collecting input arguments from command line.")
    ## Parse input arguments
    parser = argparse.ArgumentParser(
                description='Check if keyprint dataset should be output')
    ## Add argument to enable outputting keyprints
    parser.add_argument('-o', '--output',
                        help='output full dataset',
                        action='store_true')
    args = parser.parse_args()
    output = args.output ## Store output argument (default is False)


    ## Collect keyprints from user 1
    logging.info("Now collecting set of key strokes from user #1.")
    print("Now collecting set of key strokes from user #1.")
    df1 = collectData(output)
    training_prints_1 = analyze(df1)
    logging.info("Collected keyprints dataset #1.")

    time.sleep(2) ## Sleep for two seconds so user 1 can pass keyboard

    ## Collect keyprints from user 2
    logging.info("Now collecting set of key strokes from user #2.")
    print("Now collecting set of key strokes from user #2.")
    df2 = collectData(output)
    training_prints_2 = analyze(df2)
    logging.info("Collected keyprints dataset #2.")

    ## Train model
    model = train(training_prints_1, training_prints_2, output)

    ## Collect test prints in infinite loop
    while True:

        ## Take user input
        answer = input("Predict new user's identity? (Y[es] or Q[uit])")
        if answer == "Q": ## Quit program if requested
            break
        
        logging.info("Taking keyprints for prediction")
        df3 = collectData(output=False) ## Get keyprint dataset to predict on
        test_prints = analyze(df3)
        prediction, proba = predict(test_prints, model) ## Make prediction

        ## Print predicted user and probability
        print("The most recent user was {} with probability {}".format(prediction,
                                                                      round(proba,2)))

    print("Exiting...Thank you for using keyprints.")