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
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from scipy.sparse import hstack
from nltk.tokenize import RegexpTokenizer

def tokenize(text):
    """
    Tokenizer that uses porter stemmer to stemm all works
    :param text:
    :return:
    """
    text = re.sub(r'\d+', '', text) #remove numbers
    tokenizer = RegexpTokenizer(r'\w+') #remove punctuation
    tokens = tokenizer.tokenize(text)
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(item) for item in tokens]
    return stemmed

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
    #keep only first 500 word of description
    df['description'] = df['description'].apply(lambda x: ' '.join(x.split()[:500]))
    df['title'] = df['title'].apply(lambda x: x.replace('"',""))
    df['description'] = df['description'].apply(lambda x: x.replace('"',""))
    df['claims'] = df['claims'].apply(lambda x: x.replace('"',""))
    # print(df.iloc[1,:])
    df.dropna(how='any')
    response_vector = df['mainclass']

    #prep model
    n_grams = 2
    feature_model = TfidfVectorizer(
        ngram_range=(1, n_grams),
        stop_words='english',
        lowercase=True,
        strip_accents='ascii',
        decode_error='replace',
        tokenizer=tokenize,
        norm='l2',
        min_df=5,
        max_features=25000
    )

    #create tfidf matrix
    def create_tfidfmatrix(inputcolumn, docs):
        # print(inputcolumn.tolist())
        feature_matrix = feature_model.fit(inputcolumn.tolist())
        # print('Feature matrix fit', feature_matrix.vocabulary_)
        feature_matrix_transform =feature_matrix.transform(inputcolumn.tolist()).toarray()
        # print('Feature matrix fit transform', feature_matrix.shape)
        feature_df = pd.DataFrame(feature_matrix_transform, index=docs.tolist(), columns=feature_matrix.get_feature_names())

        return feature_df

    #assign matrix for each predictor
    title_tfidf_df = create_tfidfmatrix(df['title'], df['wipenumber'])
    abstract_tfidf_df = create_tfidfmatrix(df['abstract'], df['wipenumber'])
    description_tfidf_df = create_tfidfmatrix(df['description'], df['wipenumber'])
    claims_tfidf_df = create_tfidfmatrix(df['claims'], df['wipenumber'])

    #combine tfidfs created for each column
    df_feature_vector = pd.concat([title_tfidf_df,
                    abstract_tfidf_df,
                    description_tfidf_df,
                    claims_tfidf_df], axis=1)

    print(df_feature_vector.iloc[1,:])

        #show distribution of mainclasses
        # sns.countplot(y=df['mainclass'].apply(lambda x: x[:1]))
        # plt.show()

    #create featurevectors... TFIDF for all documents??
        # ngrams level (done)
        # word level (done)
        # character level
        # syntactic ngrams?
    #maybe use word embeddings based on state of the art published embeddings
    #word count
    #character count
    #word density
    #punctuation count
    #frequency of nouns, verbs, adjectives, pronouns -> POS tags would be required for this one


    #now make adjacent matrix


    featurevector = df_feature_vector
    adjacentmatrix = [[]]
    return featurevector, adjacentmatrix

def main():
    '''
    This is the main function.
	'''
    #open files
    traindf = pd.read_csv("WIPO-alpha-train.csv", nrows=20) # for testing limit number of rows (46324 in total for taining)
    # testdf = pd.read_csv("WIPO-alpha-test.csv")  #29926 total
    # print(traindf.shape)

    #preprocess data:
    trainfeaturevector, trainadjacentmatrix = preprocess_dataframe(traindf)
    # testfeaturevector, testadjacentmatrix = preprocess_dataframe(testdf)


    print("fin")

if __name__ == '__main__':
    main()
