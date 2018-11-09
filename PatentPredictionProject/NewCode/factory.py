'''
AIT 690 | Patent Classificaiton Prediction Project | Due 11/28/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
************************************************************************************************************************************************
This is a python project which predicts the IPC classficiation of patents.



The script can be run by entering:
$
***************************************************************************************
'''
import nltk
import sys
import pandas as pd
import re
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
#from scorer import score_function


def preprocess_dataframe(df):
    '''
        (1) represent each document by a feature vector.
        (2) construct a network based on the cosine similarity between every two documents and use adjacent matrix to represent network.
        (3) what we want from this process is: feature vectors for each document and a adjacent matrix.
    '''
    featurevector = []
    adjacentmatrix = [[]]
    return featurevector, adjacentmatrix

def main():
    '''
    This is the main function.
	'''
    #open file
    traindf = pd.read_csv("WIPO-alpha-train.csv")
    testdf = pd.read_csv("WIPO-alpha-test.csv")

    #preprocess data:
    trainfeaturevector, trainadjacentmatrix = preprocess_dataframe(traindf)
    testfeaturevector, testadjacentmatrix = preprocess_dataframe(traindf)


if __name__ == '__main__':
    main()
