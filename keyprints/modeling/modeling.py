## Functions to help with training models and inference

## Import necessary modules
import os
import logging

from datetime import datetime
import statsmodels.api as sm
import numpy as np
import pandas as pd

def train(prints1, prints2, output):
    """
    This function takes two keystroke datasets as pandas dataframes
    and trains a model to associate key stroke patters with either user1
    or user 2.
    ARGS:
    --prints1: keystroke dataset from user 1
    --prints2: keystroke dataset from user 2
    RETURNS:
    --model: a trained model predicting user 1 (0) or user 2 (1)
    """
    logging.info("Adding user number variable.")
    ## Add user variable to serve as outcome or target variable
    prints1['user'] = 0
    prints2['user'] = 1

    logging.info("Combining datasets.")
    ## Concatenate prints1 and prints2 dataframes
    df = pd.concat([prints1, prints2])

    ## Export full dataset if -o flag present
    ## This will enable quicker development in a jupyter notebook
    ## rather than always creating a new dataset from scratch
    if output == True:

        ## Create output data folder if it doesn't exist
        directory = "output_data"
        if not os.path.exists(directory):
            os.makedirs(directory)

        ## Create new output file with unique timestamp
        # datetime object containing current date and time
        now = datetime.now()
        date_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        csv_name = "analyzed_output-" + date_string + ".csv"
        df.to_csv("output_data/" + csv_name)

    logging.info("Training model.")
    ## Select values for X and Y for model
    X = df.drop(columns=["user"]) ## X for input variables/predictors/independent variables
    y = df["user"] ## Y for target/outcome/dependent variable

    # Train model
    model = sm.OLS(y, X).fit()

    return model


def predict(prints, model):
    """
    Predict whether a given key stroke dataset is associated with user
    1 or user 2.
    ARGS:
    --prints: An unlabeled dataset (pandas dataframe) from an unknown user
    RETURNS:
    --A prediction of user 1 or user 2
    --An associated probability of the typer being that user
    """
    
    logging.info("Making prediction.")
    ## Do X formatting just for consistency
    X = prints 
    prediction = model.predict(X)

    ## Take mean of predictions if there are mutltiple five second cuts
    prediction = prediction.mean()

    ## Avoid prediction probabilities above 1 or below zero
    ## This was originally a problem in this method because the
    ## the method was using OLS for a probabilitic prediction
    if prediction >= 1:
        prediction = .99
    elif prediction <= 0:
        prediction = .01

    ## Round prediction to avoid appearance of excessive precision
    prediction = np.round(prediction, 2)

    ## If the outcome prediction is less than .5, this means it is more
    ## likely that user 1 is the typer, and the probability of the typer
    ## being user 1 is returned. And vice versa.
    if prediction < .5:
        logging.info("User 1 with probability: {}".format(1 - prediction))
        return "User 1", 1 - prediction
    else:
        logging.info("User 2 with probability: {}".format(prediction))
        return "User 2", prediction