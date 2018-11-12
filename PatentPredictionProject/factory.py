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
import seaborn as sns
import pandas as pd
import re
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt



def preprocess_dataframe(df):
    '''
        (1) represent each document by a feature vector.
        (2) construct a network based on the cosine similarity between every two documents and use adjacent matrix to represent network.
        (3) what we want from this process is: feature vectors for each document and a adjacent matrix.
    '''

    #convert the subclasses to lists
    df['subclasses'] = df['subclasses'].apply(lambda x: x.split("--//--"))
    #convert to lowercase
    df.iloc[:,4:7] = df.iloc[:,4:7].apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

    print(df.iloc[1,:])

    # sns.countplot(y=df['mainclass'].apply(lambda x: x[:1]))
    # plt.show()

    #create featurevectors... TFIDF for all documents??
        # ngrams level
        # word level
        # character level
        # syntactic ngrams?
    #maybe use word embeddings based on state of the art published embeddings

    #word count
    #character count
    #word density
    #punctuation count
    #frequency of nouns, verbs, adjectives, pronouns -> POS tags would be required for this one
    #




    featurevector = []
    adjacentmatrix = [[]]
    return featurevector, adjacentmatrix

def main():
    '''
    This is the main function.
	'''
    #open files
    traindf = pd.read_csv("WIPO-alpha-train.csv", nrows=20) # for testing limit number of rows (38745 in total for taining)
    # testdf = pd.read_csv("WIPO-alpha-test.csv")
    # print(traindf.shape)

    #preprocess data:
    trainfeaturevector, trainadjacentmatrix = preprocess_dataframe(traindf)
    # testfeaturevector, testadjacentmatrix = preprocess_dataframe(testdf)


    print("fin")

if __name__ == '__main__':
    main()
