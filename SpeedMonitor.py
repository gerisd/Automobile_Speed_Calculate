from CentroidTracker import CentroidTracker
from Save import save
from imutils.video import VideoStream
from datetime import datetime

import numpy as np
import time
import os
import cv2
import imutils
import dlib

#preprocess image
def preprocess(image):
	image = imutils.resize(image, width=500)

	#Remove noise 
	blur = cv2.GaussianBlur(image, (5,5), 0)

	#convert to grayscale
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

	#convert to rgb for dlib
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

	return blob, gray, rgb

#distance and time (t) are both dictionaries
#Function calculates the average speed in KMPH
#Velocity = Distance / Time
def calculate_speed(frame_width, distance, t, street_measurement=10):

	#Speed for each checkpoint
	checkpoint_speeds = []

	#Number of meters per pixel
	PixelinMeters = street_measurement / frame_width

	for i in range(2, len(distance)+1):
		#Get the difference between the distances of two checkpoints
		dist_diff = abs(distance["Checkpoint" + str(i-1)] - distance["Checkpoint" + str(i)])

		#Get the time (in hours) difference between those 2 checkpoints
		time_diff = t["t" + str(i-1)] - t["t" + str(i)]
		time_secs = abs(time_diff.total_seconds())
		time_hours = time_secs / 3600

		#Distance in meters is distance in pixels x number of meters per pixel
		distanceM = dist_diff * PixelinMeters
		
		distanceKM = distanceM / 1000 

		speed = distanceKM / time_hours

		checkpoint_speeds.append(speed)

	#return the average speed 
	return sum(checkpoint_speeds) / len(checkpoint_speeds)


#directory Path to save data and image
dirPath = r"/results"

#Path to caffe model
caffePath = r"/models/MobileNetSSD_deploy.caffemodel"

#Path to object detection model
objPath = r"/models/MobileNetSSD_deploy.protxt"

#Max speed vehicle can go (KMPH)
max_speed = 50

#minimum confidence required for prediction 
min_confidence = 0.5

#initalize centroidtracker
centroidtracker = CentroidTracker()

#Load Caffe Model
model = cv2.dnn.readNetFromCaffe(objPath, caffePath)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

#list of the classes the model was trained on
labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
"bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
"person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#initalize Video Stream and warm up
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

#Keep track of total number of frames to determine whether to do object detection or object tracking for less expensive computation
totalFrames = 0

#Initalize dlib's correlation tracker, it will be used to keep track of vehicles 
tracker = None
#List to store all the tracked vehicles
tracker_list = []

#list of bounding boxes for centroidTracker object to track
boxes = []

#dictionary containing the car ID and its position
car_pos = {} 
#dictionary containing the time for each position
car_time = {}

#checkpoints list y column location for averaged speed
ckpnt = [100, 140, 180, 220]

#Set True when object is ready to calculate its speed
FindSpeed = False   

#Set True to display image
display_image = False

while True:
	frame = vs.read()

	if frame is None:
		break

	height, width = frame.shape[:2] 

	image, gray, rgb = preprocess(frame)

	#Get start time 
	t = datetime.now()

	#Perform object detection every 4 frames, since it is computationally expensive 
	if totalFrames % 4 == 0:
		#Object Detection

		#input processed image to the pre-trained network
		net.setInput(image)

		#Get the predictions from the model of the iamge
		prediction = net.forward()

		#if a detection was found, determine its class and confidence and appropriately track it
		if len(prediction) > 0:

			for i in np.arange(0, prediction.shape[2]):
				
				idx = int(prediction[0,0,i,1])
				confidence = prediction[0,0,i,2]

				#Pass if the predictioned object is a car or the probability of it being a car is lower than the minimum confidence
				if labels[idx] != 'car' or confidence < min_confidence:
					continue

				#Car detected, so now we want to get the car's bounding box so we can perform object tracking
				(x1, y1, x2, y2) = int(prediction[0,0,i,3:7] * np.array([width, height, width, height]))
				

				#Draw bounding box rectangle around detected object and begin tracking it using dlib's correlation tracker
				tracker = dlib.correlation_tracker()
				box = dlib.rectangle(x1, y1, x2, y2)
				tracker.start_track(rgb, box)

				tracker_list.append(tracker)

	else:
		#Object Tracking

		#Loop over every tracked object stored in the tracked list
		for tracker in trackers:
			#update the tracker 
			tracker.update(rgb)

			#Get the position of the object
			position = tracker.get_position()

			#Get the bounding box
			x1 = int(position.left()), y1 = int(position.top()), x2 = int(position.right()), y2 = int(position.bottom())

			#Store into list for centroid tracking
			boxes.append((x1, y1, x2, y2))

	#CentroidTracker will track the objects using the bounding boxes and return a dictionary containing the object with an ID
	objects = centroidtracker.track(boxes)		

	#Loop over the objects and determine their speed

	for (objectID, centroid) in objects.items():

		#check if the car is already being tracked...otherwise add it to dict
		if objectID not in car_pos:
			car_pos[objectID] = {"Direction": None, "Checkpoint1": None, "Checkpoint2": None, "Checkpoint3": None, "Checkpoint4": None}
			car_time[objectID] = {"t1": None, "t2": None, "t3": None, "t4": None}

			#If the Y column of the centroid is less than checkpoint 1, car is driving towards the right
			if centroid[1] < ckpnt[0]:
				val = car_pos[objectID]
				val['Direction'] = "left_to_right"
			#If the Y column of the centroid is greater than checkpoint 4, then the car is driving towards the left
			elif centroid[1] > ckpnt[-1]: 
				val = car_pos[objectID]
				val['Direction'] = "right_to_left"

		#Access inner dictionary containing vehicle information 
		val_pos = car_pos[objectID]
		val_time = car_time[objectID]

		#If the vehicle is going from left to right, check which checkpoint it is at and update it
		if val_pos["Direction"] == "left_to_right":
			#Loop over the checkpoints from left to right and find the checkpoint that hasnt been reached yet and determine if it needs to be updated 
			for (i, (k, v)) in enumerate(val_pos.items()):
				if v == None:
					#Check if its past the checkpoint, if so, update checkpoint with the vehicle's x-position and time
					if centroid[1] >= ckpnt[i]:
						val_pos[k] = centroid[0]
						val_time["t" + str(i+1)] = time 
				#If all checkpoints have been surpassed, set boolean to true to begin calculating vehicle's speed
				else:
					FindSpeed = True

		#If the vehicle is going from right to left, check which checkpoint it is currently at and update it 
		elif val_pos["Direction"] == "right_to_left":	
			#Reverse the order of the dictionary 
			for(i, (k, v)) in enumerate(reversed(list(val_pos.items()))):
				if v == None:
					#Check if its past the checkpoint, if so, update the checkpoint with the position and time
					if centroid[1] <= ckpnt[len(ckpnt)-i]:
						val_pos[k] = centroid[0]
						val_time["t" + str(len(ckpnt)-i)] = time 
				#If none of the checkpoints are empty and all have been reached, set boolean to true to find the vehicle's speed		
				else:
					FindSpeed = True

		#If True, we calculate the vehicle's speed and if it has surpassed the maximum allowed speed, log it
		if FindSpeed:

			speed = calculate_speed(width, val_pos, val_time)

			if speed > max_speed:
				#Draw a circle at centroid and ID marking the vehicle
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
				cv2.putText(frame, f"ID: {objectID}", (centroid[0], centroid[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

				currDate = t.strftime("%m-%d-%y")

				save(dirPath, frame, objectID, t, currDate, speed)

			#Reset Boolean
			FindSpeed = False

	#If boolean is true, display image (press q to quit)
	if display_image:
		cv2.imshow("Vehicle Monitor", frame)

		key = cv2.waitKey(1) & 0xFF

		#Quit when q is pressed
		if key == ord('q'):
			break	

	totalFrames += 1

cv2.destroyAllWindows()
vs.stop()
