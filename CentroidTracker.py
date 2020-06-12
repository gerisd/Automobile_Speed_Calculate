from collections import OrderedDict
from scipy.spatial import distance 
import numpy as np

class CentroidTracker:

	def __init__(self):
		#ID for each object using a dict with structure ID: centroid
		self.objects = OrderedDict()
		self.id_counter = 0

		#initalize ID list - available IDs after removal (so we dont have ID 10 when 2 is available)
		self.avail_ID = [] 

		#max distance two centroids can be, before we consider them different objects
		self.max_distance = 50

		#number of frames to check if an object has disappeared
		self.max_disappeared = 50
		#list containing the ID of the disappeared object {ID: centroid}
		self.disappeared = OrderedDict()

	#box is a list of coords (x1,y1,x2,y2)
	def track(self, boxes):
		#list containing bounding box centroids
		centroids = []

		#if there are no objects being tracked, then all the incoming bounding boxes are new objects
		if len(objects) == 0:
			for box in boxes:
				(x1, y1, x2, y2) = box
				#x pos of centroid is the width of the box / 2
				cent_X = int((x2 - x1) / 2.0)
				#y pos of centroid is the height of the box / 2
				cent_Y = int((y2 - y1) / 2.0)

				#add centroids as objects 
				add_object((cent_X, cent_Y))
				
		#if objects is not empty, then we need to compare the centroids of the new boxes and existing objects
		#the closest centroid is the same object 		
		else:
			for box in boxes:
				(x1, y1, x2, y2) = box
				#x pos of centroid is the width of the box / 2
				cent_X = int((x2 - x1) / 2.0)
				#y pos of centroid is the height of the box / 2
				cent_Y = int((y2 - y1) / 2.0)

				#temporarily store new boxes into list for comparision
				centroids.append((cent_X, cent_Y))

			#compare centroids and the closest two points are considered the same object
			euc_dist = distance.cdist(self.objects.values(), centroids)

			rows = euc_dist.min(axis=1).argsort()
			cols = euc_dist.argmin(axis=1)[rows]

			#keep track of rows and cols we already checked
			checkedRows = []
			checkedCols = []

			objectIDs = self.objects.keys()
			objectCentroids = self.objects.values()

			for (row, col) in zip(rows, cols):

				#check if the row or col has already been used
				if row in checkedRows or col in checkedCols:
					continue

				#check if the minimum distance acquired is larger than the max distance 
				#if so, it is not the same object
				if euc_dist[row, col] > self.max_distance:
					continue

				#the two centroids can be classified as the same object
				#Therefore, update (x, y) coordinates of the existing object to the new centroid
				
				#Getting the ID of the existing object
				objID = self.objectIDs[row]
				#Using the ID to access the pre-existing centroid coordinates and update them with the new coordinates
				objectCentroids[objID] = centroids[col]
				self.disappeared[objID] = 0

				#mark off the row and col so we don't check them again
				checkedRows.append(row)
				checkedCols.append(col)

			#Check if there are any remaining rows or cols that we havent used.
			#Remaining cols are coords for new objects that need to be added
			#Remaining rows are coords for existing objects have disappeared for the frame
			remainingRows = [i for i in checkedRows + rows if i not in checkedRows or i not in rows]
			remainingCols = [i for i in checkedCols + cols if i not in checkedCols or i not in cols]

			#Remaining cols are centroids for new objects, add them to the list of objects with an ID
			if len(remainingCols) > 0:
				for col in remainingCols:
					self.add_object(centroids[col])

			#Remaining rows are centroids for existing objects that have left the frame, if they are gone for too long, 
			#we remove them from our list and prepare to use its ID for the next object
			if len(remainingRows) > 0:
				for row in remainingRows:
					objID = self.objectIDs[row]
					if self.disappeared[objID] > max_disappeared:
						self.remove_object(objID)
					else:
						self.disappeared[objID] += 1
		return self.objects


	#coords in a set containing x and y for pos (x,y)
	def add_object(self, coords):
		
		#if there is a preused ID number, then use it instead of a completely new one
		if len(avail_ID) > 0:
			
			#Find index of the ID with the lowest value and store the centroid with that ID
			idx = avail_ID.index(min(avail_ID))
			lowest_ID = avail_ID[idx]
			#lowest_ID = min(avail_ID)
			objects[lowest_ID] = (coords[0], coords[1])

			#remove the lowest ID from the list 
			del avail_ID[idx]
		#There is no preused ID number, so increment counter and add a number
		else:
			objects[id_counter] = (x, y)
			id_counter += 1

	def remove_object(self, objID):
		del objects[objID]
		del disappeared[objID]
		avail_ID.append(objID)
