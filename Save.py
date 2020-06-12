#Program saves image into a folder and saves imageID, time, date, and speed into a csv file

import os
import time
import cv2
import csv

def save(dirPath, image, imageID, time, date, speed):

	csv_Path = os.path.join(dirPath, 'vehicle_speed.csv')

	#make directory if it doesn't exist
	if not os.path.exists(dirPath):
		os.mkdir(dirPath)

	#save image to given directory
	cv2.imwrite(dirPath, image)

	#Check if csv file exists, if not create, if so append
	if os.path.isfile(csv_Path):

		#save data to csv file
		with open(csv_Path, mode='w') as speed_file:
			titles = ['ID', 'date', 'time', 'speed']
			writer = csv.DictWriter(speed_file, fieldnames=titles)

			writer.writeheader()
			writer.writerow({'ID': imageID, 'Date': date, 'Time': time, 'Speed (KMPH)': speed})

	else:
		#append to the existing csv file
		with open(csv_Path, mode='a') as speed_file:
			titles = ['ID', 'date', 'time', 'speed']
			writer = csv.DictWriter(speed_file, fieldnames=titles)

			writer.writerow({'ID': imageID, 'Date': date, 'Time': time, 'Speed (KMPH)': speed})
	