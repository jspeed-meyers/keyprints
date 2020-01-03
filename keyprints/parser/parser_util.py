## keyprint data parsing functions

## Import necessary modules
import math
import logging
import numpy as np
import pandas as pd

def analyze(keys):
    """
    This function splits the keystroke dataset into five second segments and then
    creates a numerical representation of each five second segment.

    ARGS:
    --keys: the pandas dataframe of all keystrokes (up and down), 
    and timestamp
    RETURNS:
    --A list of five-second segments of keystroke converted into numeric
    retpresentations
    """

    ## Constant to hold numbers of seconds in the time window by which
    ## to split the entire keypress dataset
    WINDOW = 5

    logging.info("Beginning analyzing keys.")

    ## Find earliest (min) and latest (max) time of keypress
    try:
        time_min = math.ceil(keys.time.min())
    except ValueError as e:
        time_min = 0
        logging.warning("Did not type long enough. Error: {}".format(e))
    time_max = math.floor(keys.time.max())

    dfs = [] ## List to store windows of keypress dataset

    logging.info("Splitting keypress dataset by time windows.")

    ## Loop over time window from minimum time to maximum time
    for counter in range(time_min, time_max, WINDOW):

        upper_limit = counter + WINDOW

        ## Append only keys greater than or equal to lower limit
        ## (i.e. counter) and less than upper limit
        dfs.append(keys.loc[(keys['time'] >= counter) & 
                       (keys['time'] < upper_limit)])   
    
    data = [] ## Store final database of keypress features by time window

    ## Create dataset of keypress windows converted into features
    for df in dfs:
        features = create_features(df)
        data.append(features)

    ## Convert dataset to pandas dataframe
    data = pd.DataFrame(data,
                        columns=["a_mean",
                                 "b_mean",
                                 "c_mean"])

    return data


def create_features(keys):
    """
    Convert a keypress window into a feature representation. The current features
    use the average time and standard deviation that keys "A", "B", "C" are 
    held down (from key down to key up), though this is only a prototype and
    will likely change. To be clear, this is three separate variables.

    ARGS:
    --keys: a time window of keys (doesn't have to be any particular time length)
    These keys must be a pandas dataframe.
    RETURNS:
    --A list of numeric features for a given keypress dataset time window. There
    will be two numbers (mean key hold time and standard deviation of key hold
    time) for each key (e.g. "A", "B", ...) selected.
    """
    
    logging.info("Creating features for a keypress time window.")

    features = []

    keys = keys.copy()

    ## Loop through specified letters, create a new column of 0/1
    ## for each specified letter, find time difference between key
    ## down and key up within each letter column, keep only rows
    ## associated with key release because this captures the total
    ## key held time (note the use of diff), and the calculate
    ## key hold mean and standard deviation.
    ## NOTE: to use all upper case letters: string.ascii_uppercase
    for letter in "AEI":
        keys[letter] = np.where(keys.key == letter, 1, 0)
        letter = keys.loc[keys[letter] == 1].copy()
        letter['delta'] = letter.loc[:,'time'].diff()
        letter = letter[letter.stroke == 'release'].copy()
        features.append([letter.delta.mean()])

    features = np.concatenate(features)

    ## Convert any NAN's to 0's
    features = np.nan_to_num(features)

    return features