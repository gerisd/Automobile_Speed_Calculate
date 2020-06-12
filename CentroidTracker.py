'''
HOBBYIST BUNDLE CHAPTER 19

				CENTROID TRACKING ALGORITHM

STEP 1:	ACCEPT BOUNDING BOX COORDINATES AND COMPUTE CENTROIDS
STEP 2: COMPUTE EUCLIDEAN DISTANCE BETWEEN NEW BOUNDING BOXES AND EXISTING OBJECTS
STEP 3: UPDATE (x, y) COORDINATES OF EXISTING OBJECTS
STEP 4: REGISTER NEW OBJECTS
STEP 5: DEREGISTER OLD/LOST OBJECTS THAT HAVE MOVED OUT OF FRAME

'''
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
		#Step 1: Compute centroids (x,y) from bounding boxes
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

			'''
			EXPLANATION:
			
			cdist does euclidean distance for every set of (x,y) coords in both matrices
			cdist will return a matrix containing all distances, 
			where the # of rows is number existing objects, and # of rows is the number of new bounding boxes
			
			EX: so if we have a 2 existing objects (therefore (2,2), each containing (x,y)), and 3 new centroids (3,2)
			then euc_dict would return a (2,3) matrix. 
			The entire first row is every new centroid compared to the first existing object, where each column is one of the new centroids
			and the second row is the existing object compared to every new centroid, each compared euclidean distance is a column, hence 3 columns 


			for each existing object, get need to get the smallest euc_dist for each row 

			Using the smallest euclidean distance, there will be 3 conditions:
			
			1 - if it is within a certain range, we can acknowledge that distance between the two centroids is small enough to consider them
			the same object
			Therefore we need to update the (x, y) coords of the existing object to the new centroid

			2 - if the euclidean distance is too large then we acknowledge that this is a new object

			3 - if an existing object has no updated coords, we can ackowledge that the object has disappeared
			'''

			#compare centroids and the closest two points are considered the same object
			euc_dist = distance.cdist(self.objects.values(), centroids)

			rows = euc_dist.min(axis=1).argsort()
			cols = euc_dist.argmin(axis=1)[rows]

			#This is better, it assigns the new centroids to its closest existing object's centroid, instead of vice versa
			#This avoids failing to assign a tracked centroid when two tracked centroids have the same input centroid as their closest pairing.
			#cols = euc_dist.min(axis=0).argsort()
			#rows = euc_dist.argmin(axis=0)[cols]

			#keep track of rows and cols we already checked
			checkedRows = []
			checkedCols = []

			#Convert the IDs and coords into lists because once I have the minimum distance between 2 centroids 
			#I can use the given row and col to access the correct row in the ID's (unable to use the row in a dict)
			#likewise, I can use the col to access the correct new centroid 
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

				#print(f"row: {row} and col: {col}")
				#print(f"old centroid is {objectCentroids[objID]}, new centroid is: {centroids[col]}") 

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
