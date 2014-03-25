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


data_path='D:/Kaggle/loan default prediction/'
trainFile = 'D:/Kaggle/loan default prediction/train_v2.csv'
cleanTrainFile='D:/Kaggle/loan default prediction/train_v2noNA.csv'
testFile='D:/Kaggle/loan default prediction/test_v2.csv'
filename='D:/Kaggle/loan default prediction/test.csv'
tempPath='D:/Kaggle/loan default prediction/temp_em/'
createCV=True
categorical_vars=['f776','f777','f778']
labels='loss'

# According to this the best clustering unsurprisingly is simply k=2
# funs.missingEM(trainFile,data_path,10,categorical_vars,labels,kfolds=2)
#Missing data was already stored simply use the new data files
#Don't forget that I still need to change the categorical variables into dummies for the regression

# meanSils=pickle.load(open(data_path+'/'+'meanSils','rb'))

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

 