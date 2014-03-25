'''
Created on Feb 22, 2014

@author: julian
'''
import numpy as np
import dataSearchUtils as dSu
import dataAnalysis as dA
import os
import pickle as pickle
dataPath='D:/research/Activity recognition study/FeatureDataset'
dataCols=['avg','std']
labelsCol='Label'
discard=[]
extension='.dat'


cleaningIds=False
# 7    standing    sitting    walking    biking    bending    lying    falling
activity='0'

# This was run already an now the data includes the missing data separated from the 
# data
# #Creating new files with a reduced set of features and transformed into python data structures
# filesList=dA.multipleDataLoad(dataPath, dataCols, discard, extension,labelsCol)

#Now reading all the modified files and using them for creating the histograms
filesList=os.listdir(dataPath)
filesList=[i for i in filesList if i.find('.pyd')>=0]
print(filesList)
  
if cleaningIds: 
    for file in filesList:
        filename=dataPath+'/'+file
        dataset=pickle.load(open(filename,'rb'))
        print(np.shape(dataset['data']))
        print(np.shape(dataset['Ids']))
        Ids=dataset['Ids']
         
    for file in filesList:
        Inds=[]
        filename=dataPath+'/'+file
        dataset=pickle.load(open(filename,'rb'))
        for i in range(len(dataset['Ids'])):
            try :
                if Ids.index(dataset['Ids'][i]):
                    continue
            except:
                Inds.append(i)
        for i in range(len(Inds)-1,-1,-1):
            dataset['data']=np.delete(dataset['data'],Inds[i],1)
        dataset['Ids']=Ids
         
        indsMissing=dSu.listStrFind(dataset['header'], 'm_')
        indsMissing=dSu.findMultiple(dataset['Ids'], indsMissing)
         
        if Inds!=[]:
            pickle.dump(dataset,open(filename,'wb'))
    print('Done modifying the data files')
             
         
             
for file in filesList:
    filename=dataPath+'/'+file
    dataset=pickle.load(open(filename,'rb'))
    inds=dSu.find(dataset['labels'], lambda x:x==activity)
    hist=dA.partialhist()
    
    
    func=hist.fit(dataset['data'][inds,:],100)
    print('here')
     
 
 
     
# hist=dA.partialhist()
# func=hist.fit
 
