# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:10:36 2018

@author: gxjco
"""
import numpy as np
from scipy.spatial.distance import cosine
import random

def find_node():
    train=np.load('train.npy')
    test=np.load('test.npy')
    train_label=np.load('train_label.npy')
    test_label=np.load('test_label.npy')  
    features=np.concatenate((train,test),axis=0)
                           
    labels=np.concatenate((train_label,test_label),axis=0)
    labels=list(labels)
    label_list=list(set(labels))  #convert label to hot vector for each sample
    new_label_list=[]
    for l in label_list:
       if labels.count(l)>=4:
           new_label_list.append(l)
         
    new_labels=[]
    new_features=[]
    for i in range(len(labels)):
        if labels[i] in new_label_list:
            new_labels.append(labels[i])
            new_features.append(features[i])            
    new_features=np.array(new_features)
    
    return new_features.shape[0]
        
        
def read_data(self):    
    train=np.load('train.npy')
    test=np.load('test.npy')
    train_label=np.load('train_label.npy')
    test_label=np.load('test_label.npy')  
    features=np.concatenate((train,test),axis=0)
                           
    labels=np.concatenate((train_label,test_label),axis=0)
    labels=list(labels)
    label_list=list(set(labels))  #convert label to hot vector for each sample
    new_label_list=[]
    for l in label_list:
       if labels.count(l)>=4:
           new_label_list.append(l)
         
    new_labels=[]
    new_features=[]
    for i in range(len(labels)):
        if labels[i] in new_label_list:
            new_labels.append(labels[i])
            new_features.append(features[i])
            
    new_features=np.array(new_features)
    new_node=new_features.shape[0]
                    
    num_class=len(new_label_list)
    label=np.zeros((new_node,num_class))  
    for i in range(new_node):
        j=new_label_list.index(new_labels[i])
        label[i][j]=1
    #shuffle
    s = np.arange(new_features.shape[0])
    np.random.shuffle(s)
    f=new_features[s]
    l=label[s]
      
    adj = np.zeros((new_node,new_node))   #compute adjacent matrix
    for i in range(new_node):
        for j in range(i,new_node):
            if cosine(f[i],f[j])>0.98:
                adj[i][j]=1        
        
    Rr_data=np.zeros((self.No,self.Nr),dtype=float);  #transform relation format
    Rs_data=np.zeros((self.No,self.Nr),dtype=float);
    Ra_data=np.zeros((self.Dr,self.Nr),dtype=float);       
    cnt=0
    for i in range(self.No):
       for j in range(self.No):
         if(i!=j):
           Rr_data[i,cnt]=1.0;
           Rs_data[j,cnt]=1.0;
           Ra_data[0,cnt]=adj[i,j]
           cnt+=1; 
    return np.transpose(f),np.transpose(l),Ra_data,Rr_data,Rs_data

