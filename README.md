# Automobile Speed Calculator
Track vehicles and determine their approximate average velocity

## Overview
* Create a tracking class that monitors objects in frames
* Create a script that saves all the vital information about the vehicle
* Monitor vehicles and calculate their approximate average velocity, determining if they are speeding
___

## Project Components
* Dataset
	* If unable to use a Camera module, you can create a video dataset of vehicles tranversing from one end of the frame to the other 
* Save.py: Script saves frame into a folder and saves imageID, time, date, and speed into a csv file
* CentroidTracker.py: Class tracks object's centroid and determines if the object is the same in new frames    
* SpeedMonitor.py: Uses the tracking class to monitor vehicles and classify their average velocity  
