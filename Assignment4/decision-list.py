'''
AIT 690 | Assignment 4 | Due 10/30/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
************************************************************************************************************************************************
Write a Python program called decision-list.py that implements a decision list classifier to
perform word sense disambiguation. This method is described on pages 641-644 of the JM text. You
should also read the original paper about this method, which is available on BlackBoard (YarowskyDecision-List-1994.pdf).
Note that even though that paper focuses on accent restoration, the
method can be used with many classification problems, including word sense disambiguation.
Your program should use as many features from the original Yarowsky’s method as you think will result
in an accurate classifier. You are free to add other features if you think they will help. Please make sure
you only identify features from the training data, and that you clearly explain what features you are using
in your detailed comments. Your classifier should run from the command line as follows:
$ python decision-list.py line-train.xml line-test.xml my-decision-list.txt >
my-line-answers.txt
This command should learn a decision list from line-train.xml and apply that decision list to each
of the sentences found in line-test.xml in order to assign a sense to the word line. Do not use
line-test.xml in any other way (and only identify features from line-train.xml). Your program
should output the decision list it learns to my-decision-list.txt. You may format your decision
list as you wish, but please make sure to show each feature, the log-likelihood score associated with it,
and the sense it predicts. The file my-decision-list.txt is intended to be used as a log file in
debugging your program. Your program should output the answer tags it creates for each sentence to
STDOUT. Your answer tags should be in the same format as found in line-answers.txt.
line-train.xml contains examples of the word line used in the sense of a phone line and a product
line where the correct sense is marked in the text (to serve as an example from which to learn). linetest.xml
contains sentences that use the word line without any sense being indicated, where the
correct answer is found in the file line-answers.txt. You can find line-train.xml and linetest.xml
in the files section of our site in a compressed directory called line-data.zip.
Your program decision-list.py should learn its decision list from line-train.xml and then
apply that to line-test.xml.
You should also write a utility program called scorer.py which will take as input your sense tagged
output and compare it with the gold standard "key" data in line-answers.txt. Your scorer program
should report the overall accuracy of your tagging, and provide a confusion matrix similar to the one found
on page 156 of JM. This program should write output to STDOUT.
The scorer program should be run as follows:
$ python scorer.pl my-line-answers.txt line-answers.txt
You can certainly use your scorer.py program from the previous assignment as a foundation for this
program.
Both decision-list.py and scorer.py should be documented according to the standards of the
programming assignment rubric. In decision-list.py include your accuracy and confusion matrix
in the comments. And compare your results to that of the most frequent sense baseline.
Please submit your program source code (decision-list.py and scorer.py) as well as a script
file called decision-list-log.txt that you should create as follows:
$ script decision-list-log.txt
$ python decision-list.py line-train.xml line-test.xml my-decision-list.txt >
my-line-answers.txt
$ head -50 my-decision-list.txt
$ head -10 my-line-answers.txt
$ python scorer.py my-line-answers.txt line-answers.txt
$ exit
***************************************************************************************
'''
import nltk
import sys
import re
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
from xml.dom.minidom import parse
import xml.dom.minidom
from nltk.tokenize import word_tokenize
import math



def process_train_text(filename):
    'This function is used to read and transform the input text into a usable form'
    'Each item is an instance, each instance has a answer sense and some sentences'
    #collect xml dom structure components
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    instances=collection.getElementsByTagName("instance")
    contexts=collection.getElementsByTagName("context")
    answers=collection.getElementsByTagName("answer")
    new_text=defaultdict(str) #craete otuput dictionary
    #assign dom elements to "newtext" dictionary containing context and answers
    for i in range(len(instances)):
        id_=instances[i].getAttribute("id")
        new_text[id_]=defaultdict(str)
        new_text[id_]['answer']=answers[i].getAttribute("senseid")
        s=contexts[i].getElementsByTagName("s")
        new_text[id_]['context']=[]
        for j in range(len(s)):
            if_word=s[j].getElementsByTagName("head")
            if if_word!=[] and if_word[0].firstChild.data in ['line','lines']:
              new_text[id_]['context'].append(s[j].childNodes[0].data+'line'+s[j].childNodes[2].data)
    return new_text


def collect_training_context(new_text,K):
    'This function collects all the k patterns of context words from train text and return a pattern list and label list'
    'This function also extract the collocational features from all the patterns'
    'six features are：-1W,+1W,+1W+2W,-1W-2W,+kW,-kW)'
    word='line' #target word
    pattern_list=[] #list of sentences with all words in conext including target word
    label_list=[] #list of sentences with tagged sense
    for id_ in new_text:
        instance=new_text[id_]
        for sentence in instance['context']:
            sentence=sentence.replace('lines','line')
            word_token=word_tokenize(sentence)
            if word in word_token:
              #word_index=word_token.index(word)
              pattern_list.append(word_token)
              label_list.append(instance['answer'])

    labels=list(set(label_list))#find all possible senses

    #create Yarowsky's decision features
    f_1W=defaultdict(str) # -1word from target
    f_W1=defaultdict(str)# +1word from target
    f_1W2W=defaultdict(str)# -1 and -2 words from target
    f_W1W2=defaultdict(str)# +1 and +2 words from target
    f_KW=defaultdict(str)# -K words from target
    f_WK=defaultdict(str)# +K words from target

    #gather frequency that each pattern occurs in training data with given accent:
    for pattern in pattern_list: #for each sentence
         idx=pattern_list.index(pattern) #sentence id
         index=pattern.index(word) # id for target word in sentence


         #assign to -1word feature frequency
         try:
             if ' '.join(pattern[index-1:index+1]) in f_1W: # if -1word, target word is already stored, add 1 to freq
                 f_1W[' '.join(pattern[index-1:index+1])][label_list[idx]]+=1
             else:
                 f_1W[' '.join(pattern[index-1:index+1])]=defaultdict(str) #else store that entry
                 for l in labels:
                     f_1W[' '.join(pattern[index-1:index+1])][l]=0.1 # set all possible senese to 0.1 for smoothing
                 f_1W[' '.join(pattern[index-1:index+1])][label_list[idx]]+=1 #add 1 to freqency
         except: None
         #assign to +1word feature frequency
         try:
             if ' '.join(pattern[index:index+2]) in f_W1:
                 f_W1[' '.join(pattern[index:index+2])]+=1
             else:
                 f_W1[' '.join(pattern[index:index+2])]=defaultdict(str)
                 for l in labels:
                     f_W1[' '.join(pattern[index:index+2])][l]=0.1
                 f_W1[' '.join(pattern[index:index+2])][label_list[idx]]+=1
         except: None

         #assign to -1word, -2word feature frequency
         try:
             if ' '.join(pattern[index-2:index+1]) in f_1W2W:
                 f_1W2W[' '.join(pattern[index-2:index+1])]+=1
             else:
                 f_1W2W[' '.join(pattern[index-2:index+1])]=defaultdict(str)
                 for l in labels:
                     f_1W2W[' '.join(pattern[index-2:index+1])][l]=0.1
                 f_1W2W[' '.join(pattern[index-2:index+1])][label_list[idx]]+=1
         except: None

         #assign to +1word, +2word feature frequency
         try:
             if ' '.join(pattern[index:index+3]) in f_W1W2:
                 f_W1W2[' '.join(pattern[index:index+3])]+=1
             else:
                 f_W1W2[' '.join(pattern[index:index+3])]=defaultdict(str)
                 for l in labels:
                     f_W1W2[' '.join(pattern[index:index+3])][l]=0.1
                 f_W1W2[' '.join(pattern[index:index+3])][label_list[idx]]+=1
         except: None

         #assign to -K word feature frequency
         try:
             if ' '.join(pattern[index+K]) in f_WK:
                 f_WK['line_'+''.join(pattern[index+K])]+=1
             else:
                 f_WK['line_'+''.join(pattern[index+K])]=defaultdict(str)
                 for l in labels:
                   f_WK['line_'+''.join(pattern[index+K])][l]=0.1
                 f_WK['line_'+''.join(pattern[index+K])][label_list[idx]]+=1
         except: None

         #assign to +K word feature frequency
         try:
             if ' '.join(pattern[index-K])+'_line' in f_KW:
                 f_KW[''.join(pattern[index-K])+'_line']+=1
             else:
                 f_KW[''.join(pattern[index-K])+'_line']=defaultdict(str)
                 for l in labels:
                     f_KW[''.join(pattern[index-K])+'_line'][l]=0.1
                 f_KW[''.join(pattern[index-K])+'_line'][label_list[idx]]+=1
         except: None

         return f_1W,f_W1,f_1W2W,f_W1W2,f_KW,f_WK,labels


def pattern_likelyhood(f_1W,f_W1,f_1W2W,f_W1W2,f_KW,f_WK,labels):
    'This fucntion counts the patterns and compute the likelyhood of each pattern as well as the label infered by this pattern'
    pattern_log=defaultdict(str)
    feature_name=[f_1W,f_W1,f_1W2W,f_W1W2,f_KW,f_WK]
    print(labels)
    for feature in feature_name: #for each feature
      for pattern in feature: #for each sentence
         pattern_log[pattern]=defaultdict(str)
         log_list=[feature[pattern][labels[0]],feature[pattern][labels[1]]]
         label_idx=log_list.index(max(log_list)) #find the infered label from this pattern
         pattern_log[pattern]['value']=abs(math.log(feature[pattern][labels[0]]/feature[pattern][labels[1]])) #assign the conditional probability ratio
         pattern_log[pattern]['label']=labels[label_idx] #assign the label
    return pattern_log


def generate_list(pattern_log):
    '''This function generate the my-decision-list.txt based on the strongest pattern
    The txt form is ---pattern/label/log-score----
    This will be used when assigning values to test case
    '''
    log_list=[]
    pattern_list=[]
    sorted_pattern=[]
    for pattern in pattern_log:
        log_list.append(pattern_log[pattern]['value'])
        pattern_list.append(pattern)
    for i in range(len(log_list)):
        idx=log_list.index(max(log_list))
        sorted_pattern.append(pattern_list[idx])
        if log_list[idx]>2.4:
          with open("my-decision-list.txt", "a+") as text_file:
             text_file.write(pattern_list[idx]+"\\"+pattern_log[pattern_list[idx]]['label']+'\\'+str(pattern_log[pattern_list[idx]]['value'])+'\n')
          del log_list[idx]
          del pattern_list[idx]
        else: break


def process_test_text(filename):
    'This function input and tranform the test file into desired format'


def test(testText,filename):
   'This function use the decision list to do the wsd for each words in test, and generate the my-line-answers.txt'



def main():
    '''
    This is the main function.
	'''
    #training
    trainText = process_train_text(sys.argv[1])
    f_1W,f_W1,f_1W2W,f_W1W2,f_KW,f_WK,labels=collect_training_context(trainText,3)
    pattern_log=pattern_likelyhood(f_1W,f_W1,f_1W2W,f_W1W2,f_KW,f_WK,labels)
    generate_list(pattern_log)

    #testing
    testText = process_test_text(sys.argv[2])
    test(testText,sys.argv[3])


if __name__ == '__main__':
    main()
