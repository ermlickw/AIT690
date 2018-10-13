'''
AIT 690 | Assignment 3 | Due 10/17/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
************************************************************************************************************************************************
Write a python program called tagger.py which will take as input a training file containing part of
speech tagged text, and a file containing text to be part of speech tagged. Your program should implement
the "most likely tag" baseline.
Note that this assignment is based on problem 5.6 from page 171 of JM. For each word in the training
data, assign it the POS tag that maximizes p(tag|word). Assume that any word found in the test data but
not in training data (i.e. an unknown word) is an NN and find the accuracy of your most likely tagger on a
given test file. Record that accuracy in your overview comment, and then add at least 5 rules to your
tagger and see how those rules affect your accuracy. Make certain to also include the rules you add and
the resulting accuracy in the overview comment as well.
The input for this assignment is found in the files section of the web site (PA3.zip). The training data is
pos-train.txt, and the text to be tagged is pos-test.txt. There is also a gold standard (manually
tagged) version of the test file found in pos-test-key.txt that you will use to evaluate your tagged
output.
Here's an example of how your tagger.py program should be run from the command line. Note that
your program output should go to STDOUT, so the file name used below could be anything. This program
will learn the most likely tag from the training data, and then tag the test file based on that model.
$ python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt
Note that your tagger should not modify pos-test.txt in any way, and that the output of the program
should make certain to handle each tagged item in the test data. You will note that in both the training
and test data phrases are enclosed in brackets [] - those indicate phrasal boundaries, and you may ignore
these since we don't use them in POS tagging.
You should also write a utility program called scorer.py which will take as input your POS tagged
output and compare it with the gold standard "key" data which I have placed in the Files section of our
group (pos-test-key.txt). Your scorer program should report the overall accuracy of your tagging,
and provide a confusion matrix similar to the one found on page 156 of JM. Again, this program should
write output to STDOUT.
The scorer program should be run as follows:
$ python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-taggingreport.txt
Note that if your accuracy is unusually low (less than the most likely tag baseline) that is a sign there is a
significant problem in your tagger, and you should work to resolve that before submission.
Please do not modify any of the files found in PA3.zip. If there is some unusual situation in that text,
please ask me or the TA. Note that there are a small number of "ambiguous" tags, where two tags are
joined with an | symbol (e.g. broker-dealer/NN|JJ). In these cases, only use the first part of speech
tag and ignore the rest.
You may use code as found in the official Python documentation, Learning Python, or Programming
Python as a part of your assignments, however, this must be documented in your source code. You may
also use NTLK.
 copy of your program source code (tagger.py and scorer.py) along with a copy of
a script file called tagger-log.txt that you should create as follows:
$ script tagger-log.txt
$ time python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt
$ head -100 pos-test-with-tags.txt
$ python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-taggingreport.txt
$ cat pos-tagging-rep
***************************************************************************************
'''
import nltk
import sys
import re
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
#from scorer import score_function

def cleanfile(TextFile):

    TextFile = re.sub(r'\]','',TextFile)
    TextFile = re.sub(r'\[','',TextFile)
    TextFile = TextFile.replace('\n','')
    TextFile = TextFile.replace('``', '')
    TextFile = TextFile.replace('%', '')
    TextFile = TextFile.replace('$', '')
    TextFile = TextFile.replace('\/', '')
    TextFile = TextFile.replace('--', '')
    TextFile = TextFile.replace(r'"', '')
    TextFile = TextFile = re.sub(r'or\/..','',TextFile)
    TextFile = TextFile = re.sub(r'\|..','',TextFile)
    TextFile = TextFile.lower()
    return(TextFile)

def appendStartWord(test_file_sentences):
    new_sentences = []
    for sentence in test_file_sentences:
        sentence = "<start> " + sentence
        new_sentences.append(sentence)
    return(new_sentences)

def calWordTagProbability(train_confd_WT,traintag_fd):
    '''Method to create Word on Tag probability. Also returns a list of all words in training set
    word_tag_proDic[word][tag]= dictionaryValue = freq(tag,word) / freq(tag)
    '''

    #create dictionary of all words
    word_tag_Dic = defaultdict(list)
    for word, tags in train_confd_WT.items():
        for t in tags:
            word_tag_Dic[word].append(t)


    #create dictionary word_tag_proDic which stores P(word|tag) = freq(tag,word) / freq(tag)
    word_tag_proDic= defaultdict(dict)
    for word,listoftag in word_tag_Dic.items(): 
        for tag in listoftag:
                fre_tag_and_word= train_confd_WT[word][tag]  #freq(tag,word)
                fre_tag = traintag_fd[tag]
                dictionaryValue=fre_tag_and_word/float(fre_tag	)		
                word_tag_proDic[word][tag]=dictionaryValue
    return(word_tag_proDic, word_tag_Dic )
	
def calTagTransitionProbability(train_confd_Tt,traintag_fd):
    '''Method to create Tag given previous tag probability
     tag_transtition_ProbDic[tag][previoustag]= dictionaryValue = freq(previous tag,tag) / freq(previous tag)
    '''

    #Dictionary to store tag along with its previous tags (Created from train_confd_Tt)
    tag_transtition_Dic = defaultdict(list)
    for a, listoftags in train_confd_Tt.items():
        for t in listoftags:
            tag_transtition_Dic[a].append(t)

    
    #Created a new dictionary tag_transtition_ProbDic which stores P(tag| previous tag) = freq(previous tag,tag) / freq(previous tag)
    tag_transtition_ProbDic= defaultdict(dict)
    for tag,listoftag in tag_transtition_Dic.items(): 
        for previoustag in listoftag:
                fre_tag_and_previoustag= train_confd_Tt[tag][previoustag]
                fre_previoustag = traintag_fd[previoustag]
                dictionaryValue=fre_tag_and_previoustag/float(fre_previoustag)			
                tag_transtition_ProbDic[tag][previoustag]=dictionaryValue
    return(tag_transtition_ProbDic)
	
def assign_tags(new_sentences,traintag_fd,word_tag_proDic,tag_transtition_ProbDic, word_tag_Dic, train_confd_WT):
    predictedTags = []


    #assign tags to words that only have a single tag and assign nouns to words not in training set
    for sentence in new_sentences:
        for word in sentence.split():
            if len(word_tag_Dic[word]) == 1:
                predictedTags.append([word,word_tag_Dic[word][0]])
            else:
                if len(word_tag_Dic[word]) > 1:
                    predictedTags.append([word, "BLANK"])
                else:
                    predictedTags.append([word, 'NN'])


    #assign most likely tag based on training data to words with no adjacent tags
    loop = True
    i = 1
    while loop:

        for elem,tag in enumerate(predictedTags):

            prevwordtag = predictedTags[(elem - 1) % len(predictedTags)][1]
            thisword = predictedTags[elem][0]
            thiswordtag = predictedTags[elem][1]
            nextwordtag = predictedTags[(elem + 1) % len(predictedTags)][1]

            # assign most likely tag based on training data to words with no adjacent tags
            if (prevwordtag == "BLANK" and nextwordtag == "BLANK" and thiswordtag == "BLANK"):
                predictedTags[elem][1] = list(train_confd_WT[predictedTags[elem][0]])[0] #assign most likely in training set


            if i==4: #assign most likely tag to consecutive blanks so only single blanks remain
                if (nextwordtag == "BLANK" and thiswordtag == "BLANK"):
                    predictedTags[elem][1] = list(train_confd_WT[predictedTags[elem][0]])[0] #assign most likely in training set
        i+=1
        if i==5: #this many iterations needed to kill all triple  double blanks
            loop = False

    #use probability formulas to predict tags for single blanks
    for elem, tag in enumerate(predictedTags):

        prevwordtag = predictedTags[(elem - 1) % len(predictedTags)][1]
        thisword = predictedTags[elem][0]
        thiswordtag = predictedTags[elem][1]
        nextwordtag = predictedTags[(elem + 1) % len(predictedTags)][1]

        # assign blanks based on probability functions
        if (thiswordtag == "BLANK" and list(train_confd_WT[thisword])[0] != None):
            scores = defaultdict(list)
            #select possible tag with greatest frequency
            for tag in list(train_confd_WT[thisword])[0].split():
                try:
                    scores[tag] = word_tag_proDic[thisword][tag] * \
                                                       tag_transtition_ProbDic[tag][nextwordtag]*\
                                                       tag_transtition_ProbDic[tag][prevwordtag]
                except KeyError:
                    scores[tag]=0

            predictedTags[elem][1] = max(scores.items(), key=operator.itemgetter(1))[0]


    #print(predictedTags)
    return(predictedTags)

def apply_rules(predictedTags,word_tag_proDic,word_tag_Dic):

    #print(predictedTags)

    for elem, tag in enumerate(predictedTags):
        prevwordtag = predictedTags[(elem - 1) % len(predictedTags)][1]
        thisword = predictedTags[elem][0]
        previousword = predictedTags[elem -1][0]
        thiswordtag = predictedTags[elem][1]
        nextwordtag = predictedTags[(elem + 1) % len(predictedTags)][1]

        #RULE 1: Tag a word as an adjective if its current tag is not NN or NNP AND if it is preceded by a determiner and followed by noun, this is contextual rule
        if(prevwordtag == 'DT' and nextwordtag == "NN" and thiswordtag != "NN" and thiswordtag != "NNP"):
            predictedTags[elem][1] = "JJ"

        #RULE 2: Eliminate VBN If VBD is an option when VBN|VBD follows <start>PRP
        if(thiswordtag == "PRP" and previousword == "<start>" and nextwordtag!="VBD"):
            nextword = predictedTags[elem + 1][0]
            if len(word_tag_Dic[nextword]) == 2:
                if ((word_tag_Dic[nextword][0] == "VBN" and word_tag_Dic[nextword][1] == "VBD") or (word_tag_Dic[nextword][0] == "VBD" and word_tag_Dic[nextword][1] == "VBN")):
                    predictedTags[elem + 1][1] = "VBD"
         
        #RULE 3: Tagging all "the" to determiners. "All" followed by a "determiner" is predeterminer 
        if(thisword == "the"):
            predictedTags[elem][1] = "DT"
            thiswordtag = predictedTags[elem][1]
        
        if(thiswordtag == "DT" and previousword == "all"):
            predictedTags[elem - 1][1] = "PDT"
		
        #RULE 4: Tag all the words ending with "ous" as adjectives
        if re.search(r".ous\b",thisword):
            predictedTags[elem][1] = "JJ"
    print(predictedTags)
  
def main():
    '''
    This is the main function.
	'''
    #open training file and clean text
    trainText = open(sys.argv[1]).read()
    trainText = cleanfile(trainText)

    #convert to tagged tuple
    trainText = [nltk.tag.str2tuple(t) for t in trainText.split()]


    #total counts of tags
    tag_frquencies = defaultdict(list)
    traintag_fd = nltk.FreqDist(tag for (word,tag) in trainText)	
                # traintag_fd.plot(cumulative=False) fun visualization not needed

    #create conditional table of [word] [POS] frequencies
    train_confd_WT = nltk.ConditionalFreqDist((w.lower(), t) for w, t in trainText)
                # print(train_confd_WT['set']['VBD'])

    #Method to create P(word | tag) probability dictionary and output a list of all words in training set
    word_tag_proDic, word_tag_Dic = calWordTagProbability(train_confd_WT,traintag_fd)
   
    #create conditional table of [POS] [POS-1] frequencies
    word_tag_pairs = nltk.bigrams(trainText)
    train_confd_Tt = nltk.ConditionalFreqDist((a[1], b[1]) for (a,b) in word_tag_pairs)
        # print(train_confd_Tt["NN"]["NN"])

    #Method to create P(T | T-1) probability dictionary
    tag_transtition_ProbDic = calTagTransitionProbability(train_confd_Tt,traintag_fd)	    

####

    # clean test file
    testText = open(sys.argv[2]).read()
    testText = cleanfile(testText)
    
    #Split the file into sentences and add start tag
    test_file_sentences = nltk.sent_tokenize(testText)
    new_sentences = appendStartWord(test_file_sentences)


    predictedTags = assign_tags(new_sentences,traintag_fd,word_tag_proDic,tag_transtition_ProbDic, word_tag_Dic, train_confd_WT)
    apply_rules(predictedTags,word_tag_proDic,word_tag_Dic)
    
    # new words are automatically assigned as nouns (NN)



    #evaluate performance and place in overview comment
    # score_function(predictedTags)

    #add manual expert rules to reassign POS tags  ---> 5.6 in NLTK book

    #reevaluate performance and add to overview comment
    # score_function(updatedpredictedTags)


if __name__ == '__main__':
    main()
