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
import os
import sys
import nltk
import sys
import seaborn as sns
import pandas as pd
import re
import numpy as np
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text, sequence
from nltk.corpus import stopwords
import pickle
from sklearn.naive_bayes import MultinomialNB

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble


def tokenize(txt):
    """
    Tokenizer that uses porter stemmer to stemm all words
    :param text:
    :return:
    """
    txt = re.sub(r'\d+', '', txt) #remove numbers
    tokenizer = RegexpTokenizer(r'\w+') #remove punctuation
    tokens = tokenizer.tokenize(txt)
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(item) for item in tokens]
    return stemmed

def embeddingtokenize(txt):
    """
    Tokenizer that uses porter stemmer to stemm all works
    :param text:
    :return:
    """
    # txt = ' '.join(txt)
    tokens = text.Tokenizer()
    tokens.fit_on_texts(txt)
    word_index = tokens.word_index
    return tokens, word_index

<<<<<<< HEAD
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)

=======
def train_model(train, train_label, test, test_label):
    '''This function will train the data using base model'''
    #train_label = np.load('train_label.npy')
    #train = np.load('train.npy')
    #test = np.load('test.npy')
    #test_label = np.load('test_label.npy')
	
    clf = MultinomialNB(alpha=0.01).fit(train, train_label)
	
    predicted = clf.predict(test)

    accuracy = np.mean(predicted == test_label) * 100
    print("Accuracy of Naive Bayes Model is", accuracy, "%")
>>>>>>> d49ae0a23b5bbc9271889e316983a12e0b4705aa

def preprocess_dataframe(df, numbtrainrows):
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

    # show distribution of mainclasses
    # sns.countplot(y=df['mainclass'].apply(lambda x: x[:4]))
    # plt.show()

    # print()
    # print(len(df[df['mainclass'].apply(lambda x: x[:4])=='B29C'])) ## good candiate for simplification problem
    # print((df[df['mainclass'].apply(lambda x: x[:4])=='B29C'])['mainclass'].nunique())

    #prep model
    n_grams = 1
    feature_model = TfidfVectorizer(
        ngram_range=(1, n_grams),
        stop_words='english',
        lowercase=True,
        strip_accents='ascii',
        decode_error='replace',
        tokenizer=tokenize,
        norm='l2',
        min_df=10,
        max_features=10000
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

    title_tfidf_df = None
    abstract_tfidf_df = None
    description_tfidf_df = None
    claims_tfidf_df = None
    # add word embedding vectors from gold standard paper -> 100 dimensions
    # https://hpi.de/naumann/projects/web-science/deep-learning-for-text/patent-classification.html
    # load the pre-trained word-embedding vectors
    # if os.path.isfile('embeddingsindex.pkl'):
    #     with open('embeddingsindex.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    #         embeddings_index = pickle.load(f)
    # else:
    #     #if it isn't saved...make it
    #     embeddings_index = {}
    #     j=0
    #     for i, line in enumerate(open('patent-100.vec', encoding="utf8")):
    #         try:
    #             values = line.split()
    #             embeddings_index[' '.join(tokenize(values[0]))] = np.asarray(values[1:], dtype='float')
    #             j=j+1
    #             print(j)
    #         except:
    #             continue
    #     # save the embeddings index
    #     with open('embeddingsindex.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #         pickle.dump(embeddings_index, f)
    #
    #
    # # create a tokenizer for features
    # # create a dataframe for all of the text in the title of document
    # test = df['title'] +df['abstract']+df['description']+df['claims']
    # test= test.apply(lambda x: ' '.join(tokenize(x)))
    # tokens, word_index = embeddingtokenize(test)
    #
    # # convert text to sequence of tokens and pad them to ensure equal length vectors
    # train_seq_x = sequence.pad_sequences(tokens.texts_to_sequences((test)), maxlen=70)
    #
    # # create token-embedding mapping
    # embedding_matrix = np.zeros((len(word_index) + 1, 100))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None and len(embedding_vector)==100:
    #         embedding_matrix[i] = embedding_vector


    #assign to train and test vectors and labels
    train_feature_vector = df_feature_vector.iloc[:numbtrainrows,:]
    test_feature_vector = df_feature_vector.iloc[numbtrainrows:,:]
    df_feature_vector =None
    train_response_vector = response_vector.iloc[:numbtrainrows]
    test_response_vector = response_vector.iloc[numbtrainrows:]
    response_vector = None

    #save the processed dataset
    np.save('train.npy',train_feature_vector)
    np.save('train_label.npy',train_response_vector)
    np.save('test.npy',test_feature_vector)
    np.save('test_label.npy',test_response_vector)

    return   train_feature_vector, train_response_vector, test_feature_vector, test_response_vector

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


def main():
    '''
    This is the main function.
	'''
    #open files
    traindf = pd.read_csv("WIPO-alpha-train.csv") # for testing limit number of rows (46324 in total for taining)
    testdf = pd.read_csv("WIPO-alpha-test.csv")  #29926 total

    #simplify the dataset to a representative sample for the sake of processing time
    traindf = traindf[traindf['mainclass'].apply(lambda x: x[:4])=='B29C']
    testdf = testdf[testdf['mainclass'].apply(lambda x: x[:4])=='B29C']
    combineddf = traindf.append(testdf)

    # #Document and class analysis:
    # print(traindf['mainclass'].nunique())
    # print(testdf['mainclass'].nunique())
    # df1 = traindf['mainclass']
    # df2 = testdf['mainclass']
    # print('number of mainclasses of train in test')
    # print(df1.isin(df2).value_counts())
    # print('number of mainclasses of test in train')
    # print(df2.isin(df1).value_counts())
    # print('number of unique mainclasses of test not in train')
    # print(df2[~df2.isin(df1)].nunique())
    # print('number of unique mainclasses of train not in test')
    # print(df1[~df1.isin(df2)].nunique())

    #preprocess data and create feature vectors:
    train_feature_vector, train_response_vector, test_feature_vector, test_response_vector = preprocess_dataframe(combineddf,len(traindf))

<<<<<<< HEAD
    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), train_feature_vector, train_response_vector, test_feature_vector, test_response_vector)
    print ("NB, N-Gram Vectors: ", accuracy)

    #build classifierscdxvd
    #train_model(train_feature_vector, test_feature_vector, response_vector)

=======
    #build classifiers
    train_model(train_feature_vector, train_response_vector, test_feature_vector, test_response_vector)
>>>>>>> d49ae0a23b5bbc9271889e316983a12e0b4705aa


    print("fin")

if __name__ == '__main__':
    main()
