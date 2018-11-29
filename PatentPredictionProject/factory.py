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
import time
import nltk
import sys
import seaborn as sns
import pandas as pd
import re
import numpy as np
import operator
from collections import defaultdict
import matplotlib.pyplot as plt

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import text, sequence
from nltk.corpus import stopwords
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import dill as pickle

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

def print_cm(cm, classes,
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
    # plt.show()
    fig1.savefig(title,dpi=100)

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
    df['description'] = df['description'].apply(lambda x: ' '.join(x.split()[:50]))
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
        min_df=15,
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
    #memory drop
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


    print(df_feature_vector.shape)

    # top percentile variance selection
    # selector = SelectPercentile(f_classif, percentile=80)
    # df_feature_vector = selector.fit_transform(df_feature_vector,response_vector)
    #
    # #PCA on feature_matrix
    # pca = PCA(n_components=100)
    # df_feature_vector = pca.fit_transform(df_feature_vector)

    #SVD instead -latent semantic analysis
    # SVDtrunc = TruncatedSVD(n_components=100)
    # df_feature_vector = SVDtrunc.fit_transform(df_feature_vector)

    #NZV on feature matrix
    # df_feature_vector = SelectKBest(chi2, k=int(0.05*df_feature_vector.shape[1])).fit_transform(df_feature_vector, response_vector)

    #another selection method - top percentile summation
    # selector = SelectPercentile(f_classif, percentile=80)
    # df_feature_vector = selector.fit_transform(df_feature_vector,response_vector)


    #assign to train and test vectors and labels
    df_feature_vector = pd.DataFrame(df_feature_vector)
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

def train_model(classifier, params, feature_vector_train, label, feature_vector_valid, valid_y):


    cross_val = KFold(n_splits=5, shuffle=True)
    clf = GridSearchCV(classifier, params, cv=cross_val.split(feature_vector_train), n_jobs=1)
    clf.fit(feature_vector_train, label)
    print('Grid Search Completed', clf.best_estimator_, clf.best_score_)


    predictions = clf.predict(feature_vector_valid)


    acc = metrics.accuracy_score(valid_y, predictions)
    prec = metrics.precision_score(valid_y, predictions, average='micro')
    cr = metrics.classification_report(valid_y,predictions)
    cm = metrics.confusion_matrix(valid_y,predictions)
    f1 = metrics.f1_score(valid_y,predictions, average='micro')
    return acc,prec, cr, cm, f1


def main():
    '''
    This is the main function.
	'''
    #open files
    traindf = pd.read_csv("WIPO-alpha-train.csv") # for testing limit number of rows (46324 in total for taining)
    testdf = pd.read_csv("WIPO-alpha-test.csv")  #29926 total

    # simplify the dataset to a representative sample for the sake of processing time
    traindf = traindf[traindf['mainclass'].apply(lambda x: x[:3])=='G09']
    testdf = testdf[testdf['mainclass'].apply(lambda x: x[:3])=='G09']

    # combine and select subclass
    combineddf = traindf.append(testdf)
    combineddf['mainclass'] = combineddf['mainclass'].apply(lambda x: (x[:4]).strip())
    # print(combineddf['mainclass'].head())

    #Document and class analysis:
    # df1 = traindf['mainclass'].apply(lambda x: (x[:4]).strip())
    # df2 = testdf['mainclass'].apply(lambda x: (x[:4]).strip())
    # print(df1.nunique())
    # print(df2.nunique())
    # print(len(combineddf))
    #
    # print('number of unique mainclasses of test not in train')
    # print(df2[~df2.isin(df1)].nunique())
    # print('number of unique mainclasses of train not in test')
    # print(df1[~df1.isin(df2)].nunique())


    #preprocess data and create feature vectors OR load created data:
    load_data = True

    if not(load_data and os.path.isfile('train.npy') and os.path.isfile('train_label.npy') and os.path.isfile('test.npy') and os.path.isfile('test_label.npy')):
        train_feature_vector, train_response_vector, test_feature_vector, test_response_vector = preprocess_dataframe(combineddf,len(traindf))
    else:
        train_feature_vector = np.load('train.npy')
        train_response_vector = np.load('train_label.npy')
        test_feature_vector = np.load('test.npy')
        test_response_vector = np.load('test_label.npy')


    classifiers = {
            'Bayes': [MultinomialNB(), {'alpha': np.arange(0.0001, 0.2, 0.0001)}],

            'SGD': [SGDClassifier(n_iter=8, penalty='elasticnet'), {'alpha':  10**-6*np.arange(1, 15, 2),'l1_ratio': np.arange(0.1, 0.3, 0.05)}],

            'Passive Aggressive': [PassiveAggressiveClassifier(loss='hinge'), {}],

            'Perceptron': [Perceptron(), {'alpha': np.arange(0.00001, 0.001, 0.00001)}],

            'LogisticRegression': [LogisticRegression(solver='lbfgs', multi_class='multinomial'), {}],

            'LDA': [LinearDiscriminantAnalysis(solver='svd'), {}],

            'QDA': [QuadraticDiscriminantAnalysis(), {}],
        }

    model = 'LogisticRegression'
    accuracy, prec, cr, cm, f1 = train_model(classifiers[model][0], classifiers[model][1],
                                                        train_feature_vector, train_response_vector,
                                                        test_feature_vector, test_response_vector)
    print (model,"|   Accuracy:", accuracy, "|  Micro-averaged Precision:", prec, "| Micro-Averaged F1 Score: ", f1)
    print("Classification Report: \n",cr)
    print("Confusion_Matrix: \n",cm)
    labels = list(set(combineddf['mainclass']))
    then=time.time()
    print_cm(cm, labels,True,model)
    print(model," finished in ",then-now, "seconds")



if __name__ == '__main__':
    now=time.time()
    main()
    then=time.time()
    print("script finished in ",then-now, "seconds")
