## Standard libraries
import logging
import os
import time

## Suppress pygame message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

## External libraries
import pygame ## key module for keystroke measurement
import numpy as np
import pandas as pd
from datetime import datetime

def collectData(output):
    """
    This is the main loop to capture keystroke dynamics, specifically
    key up and key down time stamps.

    ARGS: None
    RETURN: a pandas dataframe containing keystroke dataset
    --key up or down
    --key
    --time stamp
    """

    ## Collect output variable from function definition
    output = output

    rows = []

    # initializes Pygame
    logging.info("Initiating pygame")
    pygame.init()
    logging.info("Finished initiating pygame.")
    
    # sets the window title
    pygame.display.set_caption(u'Type')

    # sets the window size
    pygame.display.set_mode((400, 400))

    logging.info("Entering typing loop.")
    print("""PLEASE TYPE THE PHRASES:
    The cat walked to the grocery store and bought a kiwi.
    The pig walked to the book store and bought a tribe book.
    The cat then returned home and ate the kiwi, leaving trash and bowls everywhere.
    The pig came come and cleaned the dishes.
    The pig threw away the trash.
    And the pig and cat were happy!
    Oh, also please type a secret message too.
    Please type: aeiou, aeiou, aeiou. You are now an expert on vowels!
    """)

    # infinite loop
    while True:

        # gets a single event from the event queue
        event = pygame.event.wait()

        # if the 'close' button of the window is pressed
        if event.type == pygame.QUIT:

            logging.info("Quit typing loop command entered.")

            # stops the application
            df = pd.DataFrame(data=rows, 
                              columns=['stroke', 'key', 'time'])


            ## Export full dataset if -o flag present
            ## This will enable quicker development in a jupyter notebook
            ## rather than always creating a new dataset from scratch
            if output == True:

                ## Create output data folder if it doesn't exist
                directory = "output_data"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                ## Create new output file with unique timestamp
                ## datetime object containing current date and time
                now = datetime.now()
                date_string = now.strftime("%d-%m-%Y-%H-%M-%S")
                csv_name = "raw_user_output-" + date_string + ".csv"
                df.to_csv("output_data/" + csv_name)

            return df

        # captures the 'KEYDOWN' and 'KEYUP' events
        if event.type in (pygame.KEYDOWN, pygame.KEYUP):

            # gets the key name
            key_name = pygame.key.name(event.key)

            # converts to uppercase the key name
            key_name = key_name.upper()

            logging.info("Key event recorded: {}".format(key_name))

            # if any key is pressed
            if event.type == pygame.KEYDOWN:
                # prints on the console the key pressed
                rows.append(["press", key_name, time.time()])

            # if any key is released
            elif event.type == pygame.KEYUP:
                # prints on the console the released key
                rows.append(["release", key_name, time.time()])

    # finalizes Pygame
    pygame.quit()