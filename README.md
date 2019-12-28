# Keyprints
Ever wanted to try and determine user identity based on keystroke patterns? This is a sample project with the basics in place.

The name of the project is inspired by the question of whether keystroke patterns can function as the equivalent of a fingerprint, a unique identifier.

## Technical Notes

Keystrokes are captured using the pygame module. All key presses and key releases are captured along with a timestamp and a record of which key was pressed or released.

The keystrokes are then converted into a numerical vector (features) in five second intervals. The current prototype looks at the average time between key press and key release for a, e, and i, i.e. the common vowels.

After collecting keystroke data from two users, a simple linear regression model is fit on the keystroke data.

A user then types and the program predicts which user just typed.

Does it work? I'm not sure. It's a work in progress. 
