# Automobile_Speed_Calculate
Track vehicles and determine their approximate average velocity

## Overview
* Create an image dataset that a model can be trained on
* Convert the images from the training dataset to 128-D vector
* Train multiple models and select the model with the best score and predictability
* Test classifier by using a piCamera module or a test dataset
___

## Project Components
* Dataset
	* If unable to use a Camera module, you can create a video dataset of vehicles tranversing from one end of the frame to the other 
* Save.py: Script saves frame into a folder and saves imageID, time, date, and speed into a csv file
* CentroidTracker.py: Class tracks object's centroid and determines if the object is the same in new frames    
* SpeedMonitor.py: Uses the tracking class to monitor vehicles and classify their average velocity  
