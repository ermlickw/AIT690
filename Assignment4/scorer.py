"""
AIT 690 | Assignment 4 | Due 10/29/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
*****************************************************************************************
This code is used to evaluate the performance of the tagger performance in the metrics of accuracy.
It will generate a 'pos-taggingreport.txt' file to report the accuracy and a confusion matrix.
One could run the scorer.py like:
$ python scorer.py my-line-answer.txt line-answers.txt
"""

import sys
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
import numpy as np
from bs4 import BeautifulSoup as Soup
from sklearn.metrics import confusion_matrix

def score_function(soup,predicted_file):
    'This function compute the accuracy of the performance'
    score = 0
    with open(predicted_file) as predicted_file:
        for line in predicted_file:
            line_num, sense = line.split('\\')
            #Final all instances of answers in Gold file
            for answer in soup.findAll('answer'):
                #Fetch sense IDs and instance names
                ans_attrs = dict(answer.attrs)
                instance_name = ans_attrs[u'instance']
                senseid = ans_attrs[u'senseid']
                #Increase the score if sense Id matches in the predicted and gold file
                if(line_num.rstrip() == instance_name.rstrip()):
                    if(sense.strip() == senseid.strip()):
                        score = score + 1
        accuracy = score*100/126  #compute the accuracy
    return accuracy


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig1=plt.figure(figsize=(10,8))
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

def generate_cm(soup,predicted_file):
     'This fucntion is used to generate the confusion matrix and the tag label list'
     #Create an array of predicted senses
     predicted_senses = []
     with open(predicted_file) as predicted_file:
        for line in predicted_file:
            line_num, sense = line.split('\\')
            predicted_senses.append(sense.strip())

     #Create an array of Gold Standard word senses
     gold_standard_senses = []
     for answer in soup.findAll('answer'):
            #Fetch sense IDs and instance names
            ans_attrs = dict(answer.attrs)
            instance_name = ans_attrs[u'instance']
            senseid = ans_attrs[u'senseid']
            gold_standard_senses.append(senseid.strip())

     #Create an array of labels
     labels = []
     for i in range(len(gold_standard_senses)):
         if gold_standard_senses[i] not in labels:
             labels.append(gold_standard_senses[i])

     cm = confusion_matrix(gold_standard_senses, predicted_senses, labels)
     return cm.astype(int),labels

def main():
     'This is main fucntion of scorer.py'

     #Read the Gold File and parse through BeautifulSoup
     predicted_file = sys.argv[1]
     gold_file = sys.argv[2]

     handler = open(gold_file).read()
     soup = Soup(handler,"html.parser")

     accuracy = score_function(soup,predicted_file)

     print("Accuracy of word sense disambiguation assignment is: "+"%s" % accuracy+"\n")
     # #generate Confusion Matrix
     cm,label = generate_cm(soup,predicted_file)

     #draw the confusion matrix
     plot_confusion_matrix(cm,label,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues)


if __name__ == '__main__':
    main()
