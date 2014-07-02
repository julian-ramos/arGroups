import os
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import  scipy.stats as stats
import csv
import io
import pickle
import dataML_Prepro as funsPre
import dataML_Big as funsBig

#Need to change the paths so that they work both in the laptop and desktop

if os.getenv('COMPUTERNAME')=='JULIAN':
    data_path='G:\Activity_RS\Last data set\FeatureDataset/'
    #This is the file were all of the data is summarized
    trainFile = 'D:/Kaggle/loan default prediction/train_v2.csv'
    #This is the file without NAs
    cleanTrainFile='D:/Kaggle/loan default prediction/train_v2noNA.csv'
    #Testing file
    testFile='D:/Kaggle/loan default prediction/test_v2.csv'
    createCV=True
else:
    data_path='D:/Kaggle/loan default prediction/'
    trainFile = 'D:/Kaggle/loan default prediction/train_v2.csv'
    cleanTrainFile='D:/Kaggle/loan default prediction/train_v2noNA.csv'
    testFile='D:/Kaggle/loan default prediction/test_v2.csv'
    createCV=True
    
    
    
#First I have to try feature selection and with 
#a subset of the features I can then go to the next step
#However, this feature selection is for all of the population
    
    
#Here is the way I have to make this code
#Run kmeans for each individual data file
#get the centroids
#cluster the centroids
#Once that is done try to comeup with a way to 
#to do the grouping of people    



if createCV==True:
    training_files,testing_files=funsPre.crossvalidationSPF(cleanTrainFile,data_path,4)
    files=[training_files,testing_files]
    with open(data_path+'files.txt','wb') as f: 
        pickle.dump(files, f)
else:
    with open(data_path+'files.txt','rb') as f:
        training_files,testing_files=pickle.load(f)
    for i in range(len(training_files)):
        label_idx=-1
        #It seems like I have finally got the data in the right format up to this point
        #Now I need to convert the categorical variables and maybe do some other preprocessing
        #like standardizing the data and maybe just maybe some feature selection
        funsBig.partialCLF('logistic',training_files[i],testing_files[i],label_idx,categorical_vars)

 