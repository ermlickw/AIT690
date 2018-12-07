'''
AIT 690 | Patent Classificaiton Prediction Project | Due 11/28/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
************************************************************************************************************************************************
This project proposes a novel patent graphical neural network (P-GNN) approach
for the task of automated patent classification. Two experiments are performed on
the benchmarked WIPO-alpha dataset. Experiment 1 utilizes the entire data set to
make predictions at the Subclass level. Experiment 2 utilizes Section D of the dataset
to make predictions at the maingroup level.


The script can be run by entering:
$ python factory.py
$ python GNN.py


**It is recommended that you run Experiment 2 first.**
**See the Readme for information on how to run Experiment 1**

The script will output the training and testing feature vectors, the saved classifiers and the plotted confusion matricies.
It will also output performance metrics for each model via text output.

The dataset can be obtained directly from: https://drive.google.com/drive/folders/1gOBlngdaolH7OUROw3pgA02R1vEtHzM5?usp=sharing
or via the online website: https://www.wipo.int/classifications/ipc/en/ITsupport/Categorization/dataset/wipo-alpha-readme.html


***************************************************************************************
'''
#import libraries
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
from nltk.corpus import stopwords
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from numpy import loadtxt
import itertools
import numpy as np

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

# def embeddingtokenize(txt):
#     """
#     Tokenizer that uses porter stemmer to stemm all works
#     :param text:
#     :return:
#     """
#     # txt = ' '.join(txt)
#     tokens = text.Tokenizer()
#     tokens.fit_on_texts(txt)
#     word_index = tokens.word_index
#     return tokens, word_index

def print_cm(cm, classes,
              normalize=False,
              title='Confusion matrix',
              cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    fig1=plt.figure(figsize=(30,30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    fig1.savefig("CMs/"+title,dpi=100)

def preprocess_dataframe(df, numbtrainrows):
    '''
    This function cleans the dataset and represent each document by a feature vector
    The data is first cleaned, then tokenized into a TFIDF representation
    The TFIDF representations are then reduced using LSA or PCA
    The results are then saved to the directory
    '''

    #convert the subclasses to lists
    df['subclasses'] = df['subclasses'].apply(lambda x: x.split("--//--"))
    #convert to lowercase
    df.iloc[:,4:7] = df.iloc[:,4:7].apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    #keep only first 50 word of description
    df['description'] = df['description'].apply(lambda x: ' '.join(x.split()[:50]))
    df['title'] = df['title'].apply(lambda x: x.replace('"',""))
    df['description'] = df['description'].apply(lambda x: x.replace('"',""))
    df['claims'] = df['claims'].apply(lambda x: x.replace('"',""))
    df.dropna(how='any')
    response_vector = df['mainclass']

    #prep model
    n_grams = 3
    feature_model = TfidfVectorizer(
        ngram_range=(1, n_grams),
        stop_words='english',
        lowercase=True,
        strip_accents='ascii',
        decode_error='replace',
        tokenizer=tokenize,
        norm='l2',
        min_df=1,
        max_features=2000
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

##FUTURE WORK NOT INCLUDED IN PRESENT RESULTS:
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
##END OF FUTURE WORK NOT INCLUDED IN PRESENT RESULTS:


    #feature reduction:

    # top percentile variance selection method
    # selector = SelectPercentile(f_classif, percentile=80)
    # df_feature_vector = selector.fit_transform(df_feature_vector,response_vector)

    #SVD instead -latent semantic analysis
    SVDtrunc = TruncatedSVD(n_components=100)
    df_feature_vector = SVDtrunc.fit_transform(df_feature_vector)

    #PCA on feature_matrix
    # pca = PCA(n_components=100)
    # df_feature_vector = pca.fit_transform(df_feature_vector)

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
    np.save('train-D.npy',train_feature_vector)
    np.save('train_label-D.npy',train_response_vector)
    np.save('test-D',test_feature_vector)
    np.save('test_label-D.npy',test_response_vector)

    return   train_feature_vector, train_response_vector, test_feature_vector, test_response_vector

def train_model(classifier, params,
                feature_vector_train, train_y,
                feature_vector_valid, valid_y,
                model, labels,
                load_models):
    '''
    This function trains or loads the models. For training, a 5-fold gridsearch is used
    to find the optimal hyperparameters and the resutls are outputted and saved under
    classifiers
    '''

    #if not loading models then train and save them
    if load_models == False:
        cross_val = KFold(n_splits=5, shuffle=True)
        clf = GridSearchCV(classifier, params, cv=cross_val.split(feature_vector_train), n_jobs=1)
        clf.fit(feature_vector_train, train_y)
        print('Grid Search Completed', clf.best_estimator_, clf.best_score_)

        #save model:
        pickle.dump(clf.best_estimator_, open("Classifiers/"+model+'-D', 'wb'))
    else: #load model
        clf = pickle.load(open("Classifiers/"+model+'-D','rb'))


    predictions = clf.predict(feature_vector_valid)

    #create performance metrics
    acc = metrics.accuracy_score(valid_y, predictions)
    prec = metrics.precision_score(valid_y, predictions, average='macro')
    recall = metrics.recall_score(valid_y, predictions, average='macro')
    cr = metrics.classification_report(valid_y,predictions)
    cm = metrics.confusion_matrix(valid_y,predictions, labels=labels)
    f1 = metrics.f1_score(valid_y,predictions, average='macro')

    #print out performance metrics
    print (model,"|   Accuracy:", acc, "|  Macro-averaged Precision:", prec, "| Macro-Averaged F1 Score: ", f1)
    print("Classification Report: \n",cr)
    # print("Confusion_Matrix: \n",cm)
    print_cm(cm, labels,True,model)
    return acc,prec, recall, cr, cm, f1


def main(load_data,load_models,Experiment):
    '''
    This is the main function.
    Data is read in from the directory and subsampled based on the Experiment.
    Feature vectors are either created or loaded.
    Classifiers are either trained or loaded.
    Predicted results on the test set are provided.
	'''
    #build feature vectors if missing or specified by user above
    if load_data==False or not(os.path.isfile('train.npy') or os.path.isfile('train_label.npy') or os.path.isfile('test.npy') or os.path.isfile('test_label.npy')):
        #open files
        traindf = pd.read_csv("WIPO-alpha-train.csv") # for testing limit number of rows (46324 in total for taining)
        testdf = pd.read_csv("WIPO-alpha-test.csv")  #29926 total

        #determine if experiment 1 or 2:
        if Experiment == 2: # if experiment 2 and data not loaded

            # simplify the dataset to section D
            traindf = traindf[traindf['mainclass'].apply(lambda x: x[:1])=='D']
            testdf = testdf[testdf['mainclass'].apply(lambda x: x[:1])=='D']

            # combine and select the main group level
            combineddf = traindf.append(testdf)
            combineddf['mainclass'] = combineddf['mainclass'].apply(lambda x: (x[:6]).strip())
            labels = list(set(testdf['mainclass'].apply(lambda x: (x[:6]).strip())))

        else: # if experiemnt 1 and data not loaded
            # combine and select the main group level
            combineddf = traindf.append(testdf)
            combineddf['mainclass'] = combineddf['mainclass'].apply(lambda x: (x[:4]).strip())
            labels = list(set(testdf['mainclass'].apply(lambda x: (x[:4]).strip())))

        train_feature_vector, train_response_vector, test_feature_vector, test_response_vector = preprocess_dataframe(combineddf,len(traindf))
        then=time.time()
        print("Feature and Response vectors CREATED in ",round(then-now,2), "seconds")


    else: #load the feature vectors if it is in memory already
        train_feature_vector = np.load('train-D.npy')
        train_response_vector = np.load('train_label-D.npy')
        test_feature_vector = np.load('test-D.npy')
        test_response_vector = np.load('test_label-D.npy')
        then=time.time()
        labels = list(set(test_response_vector))
        print("Feature and Response Vectors LOADED in",round(then-now,2), "seconds")



    #set classifiers and hyperparameters to be searched
    classifiers = {
            'Baseline': [DummyClassifier(strategy="stratified"), {}],
            'LogisticRegression': [LogisticRegression(solver='lbfgs', multi_class='multinomial'), {}],
            'KNN': [KNeighborsClassifier(),{ 'n_neighbors': np.arange(1,4,10)}],
            'LDA': [LinearDiscriminantAnalysis(solver='svd'), {}],
            # 'Bayes': [GaussianNB(), {}], #
            'SGD': [SGDClassifier(n_iter=8, penalty='elasticnet'), {'alpha':  10**-6*np.arange(1, 15, 2),'l1_ratio': np.arange(0.1, 0.3, 0.05)}],
            'Passive Aggressive': [PassiveAggressiveClassifier(loss='hinge'), {}],
            'Perceptron': [Perceptron(), {'alpha': np.arange(0.00001, 0.001, 0.00001)}], #
        }

    #train model for all classifiers and output results
    macroprecision = {}
    for model in classifiers.keys():
        acc, prec,recall, cr, cm, f1 = train_model(classifiers[model][0], classifiers[model][1],
                                                            train_feature_vector, train_response_vector,
                                                            test_feature_vector, test_response_vector,
                                                            model,labels,
                                                            load_models)
        then=time.time()
        macroprecision.update({model:[round(f1,2), round(prec,2),round(recall,2), round(acc,2)]})
        print(model, "finished in ", round(then-now,2)/60, "minutes")

    #print final results
    print('\n FINAL RESULTS | MACRO-F1  | MACRO-PRECISION | MACRO-RECALL | ACC')
    d_view = [ (v,k) for k,v in macroprecision.items() ]
    d_view.sort(reverse=True)
    for v,k in d_view:
        print(k,v)

if __name__ == '__main__':
    now=time.time()
    #preprocess data and create feature vectors OR load created data and Choose Experiment:
    load_data = True
    load_models = True
    Experiment = 2


    main(load_data, load_models, Experiment)

    then=time.time()
    print("script finished in ",round(then-now,2)/60, "minutes")
