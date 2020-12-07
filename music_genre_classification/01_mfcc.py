import os
import pickle
import random 
import operator
import math
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from tempfile import TemporaryFile

'''
We are using MFCC (Mel Frequency Cepstral Coefficients):
http://ismir2000.ismir.net/papers/logan_paper.pdf
'''

DATASET_PATH = "./genres/"

# Distance function 
def distance(instance1, instance2, k):
  distance = 0 
  mm1 = instance1[0] 
  cm1 = instance1[1]
  mm2 = instance2[0]
  cm2 = instance2[1]
  distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
  distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1)) 
  distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
  distance -= k
  return distance

# Function to get the distance between feature vectors and neighbors
def getNeighbors(trainingSet, instance, k):
  distances = []
  for x in range (len(trainingSet)):
    dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
    distances.append((trainingSet[x][2], dist))
  distances.sort(key=operator.itemgetter(1))
  neighbors = []
  for x in range(k):
    neighbors.append(distances[x][0])
  return neighbors
    
# Identify nearest neighbors
def nearestClass(neighbors):
  classVote = {}
  for x in range(len(neighbors)):
      response = neighbors[x]
      if response in classVote:
        classVote[response] += 1 
      else:
        classVote[response] = 1
  sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
  return sorter[0][0]
    
# Define a function for model evaluation
def getAccuracy(testSet, predictions):
  correct = 0 
  for x in range (len(testSet)):
    if testSet[x][-1] == predictions[x]:
      correct += 1
  return 1.0*correct/len(testSet)
    
# Extract features from the dataset and dump them into a binary .dat file “genres.dat”
directory = DATASET_PATH
f= open("features.dat" ,'wb')
i=0
for folder in os.listdir(directory):
  i+=1
  if os.path.isdir(os.path.join(directory, folder)):
    for file in os.listdir(os.path.join(directory, folder)):  
      (rate,sig) = wav.read(os.path.join(directory, folder, file))
      mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy = False)
      covariance = np.cov(np.matrix.transpose(mfcc_feat))
      mean_matrix = mfcc_feat.mean(0)
      feature = (mean_matrix, covariance, i)
      pickle.dump(feature, f)
f.close()

# Train and test split on the dataset
dataset = []
def loadDataset(filename, split_ratio, train_set, test_set):
  with open("features.dat", 'rb') as f:
    while True:
      try:
        dataset.append(pickle.load(f))
      except EOFError:
        f.close()
        break  
  for x in range(len(dataset)):
    if random.random() < split_ratio:
      train_set.append(dataset[x])
    else:
      test_set.append(dataset[x])  
trainingSet = []
testSet = []
loadDataset("features.dat", 0.8, trainingSet, testSet)

# Make prediction using KNN and get the accuracy on test data
leng = len(testSet)
predictions = []
for x in range (leng):
  predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 4))) 
accuracy1 = getAccuracy(testSet, predictions)
print("Accuracy: {}".format(accuracy1))


