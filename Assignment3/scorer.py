"""
AIT 690 | Assignment 3 | Due 10/17/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
*****************************************************************************************
This code is used to evaluate the performance of the tagger performance in the metrics of accuracy.
It will generate a 'pos-taggingreport.txt' file to report the accuracy and a confusion matrix.

One could run the scorer.py like:
$ python scorer.py pos-test-with-tags.txt pos-test-key.txt
"""
import nltk
import sys
import re
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
from tagger import cleanfile
import itertools
import matplotlib.pyplot as plt
import numpy as np

def score_function(predicted_tag,goldline_tag):
    'This function compute the accuracy of the performance'
    score=0
    for i in range(len(goldline_tag)): #loop all the tags to find the right predicted tag
        if predicted_tag[i][1]==goldline_tag[i][1]:
            score+=1
    accuracy=score*100/float(len(goldline_tag))  #compute the accuracy
    return accuracy


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig1=plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig1.savefig('confusion_matrix',dpi=100)

def generate_cm(predicted_tag,goldline_tag):
     'This fucntion is used to generate the confusion matrix and the tag label list'
     label=[]
     for i in range(len(goldline_tag)):
         if goldline_tag[i][1] not in label and type(goldline_tag[i][1]) is str:
             label.append(goldline_tag[i][1])  #find all the tag type and make a list
     cm=np.zeros((len(label),len(label)))  #initialize the confusion matrix
     for i in range(len(goldline_tag)):
         try:
             pred=label.index(predicted_tag[i][1])
             true=label.index(goldline_tag[i][1])
             cm[true][pred]+=1   #generate the matrix
         except: None
     return cm.astype(int),label

def main():
     'This is main fucntion of scorer.py'

     #import the result and goldline
     predicted=cleanfile(open(sys.argv[1]).read())
     goldline = cleanfile(open(sys.argv[2]).read())

     predicted_tag=[nltk.tag.str2tuple(t) for t in predicted.split()]
     goldline_tag = [nltk.tag.str2tuple(t) for t in goldline.split()]

     while ('', ':') in goldline_tag:
       goldline_tag.remove(('', ':'))
     while ('', '') in goldline_tag:
       goldline_tag.remove(('', ''))
     while ('', 'NN') in goldline_tag:
       goldline_tag.remove(('', 'NN'))
     while ('', 'JJ') in goldline_tag:
       goldline_tag.remove(('', 'JJ'))

     for i in range(len(predicted_tag)):
        if goldline_tag[i][0][0]!=predicted_tag[i][0][0]:  #remove the unmatched words
            goldline_tag.remove(goldline_tag[i])


     acc=score_function(predicted_tag,goldline_tag) #compute the accuracy
     print("Accuracy of tagger assigments is: "+"%s" % acc+"\n")
     cm,label=generate_cm(predicted_tag,goldline_tag) #generate

     plot_confusion_matrix(cm,label,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues)
     #draw the confusion matrix


if __name__ == '__main__':
    main()
