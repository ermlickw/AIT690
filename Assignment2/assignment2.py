'''
This projetc is implemented by:
Xiaojie Guo, Nidhi Mehrotra and Billy Ermlick.

This Python program called ngram.py will learn an N-gram language model from an arbitrary number of plain text files.
The program can generate a given number of sentences based on that N-gram model.
This program can work for any value of N, and output m sentences as the user requires. Your can run the program as follows:
   ngram.py n m input-file/s
n refers to the number of grams and m refers to the number of sentences you want to generate.
for example:
   ngram.py 3 10 'austen-emma.txt' 'austen-persuasion.txt'
The .txt files used in this project are from <http://www.gutenberg.org>. Thus, you could chose the files name as follows:
   'austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt',
   'burgess-  busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt',
   'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt',
   'shakespeare-hamlet.txt',   'shakespeare-macbeth.txt', 'whitman-leaves.txt'

Some of the code for fetching the file and calculating Conditional Frequency Distribution is picked up from NTLK Book.
https://www.nltk.org/book/

This program will generate ngram-log.txt file in the same folder from where you run the program.
Log will be written in this format in the text file

4.443262577056885 secs python ngram.py 4 10 austen-emma.txt austen-persuasion.txt carroll-alice.txt
'''
import nltk
import sys
import random
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist,ConditionalFreqDist,ConditionalProbDist
from collections import defaultdict
from nltk.corpus import gutenberg
import time
def read_files(ngramModel, numFiles):
    '''
    This function fetches text from all the files provided in the arguments and merges the files.
    This function returns tokens as the output
    '''
    MergedText = []
    for file in range(3,numFiles):
        #Read all the files
        word =gutenberg.words(sys.argv[file])
        MergedText.extend(word)   #add

    return MergedText

def boundaries(nngrams):
    ''' delete the cross boudary nngrams'''
    new_nngrams=[]
    for i in nngrams:
         if not ( ('end_' in i and i.index('end_')!=len(i)-1) or ('start_' in i and i.index('start_')!=0)):
            new_nngrams.append(i)
    return new_nngrams

def delete_short(words,ngramModel):
    '''1. delete the words from short sentence where words amount is less than n so that they will not be considered while generating ngram model
     2. add the start and end tags on the words list so that each sentence will begin with 'start_' and end with 'end_'''
    boundary = ['.','?','!']  #put the start and end tags when find the end punctuations
    new_words=[]
    sentence=[]
    for i in words:
        sentence.append(i.lower())
        if i in boundary:
            if len(sentence)>=ngramModel:
                 sentence.append('end_')
                 new_words.extend(sentence)
            sentence=['start_']
    return new_words

def generateModel(MergedText, ngramModel, numSentences):
    '''
    This function discards the sentences where number of words are less than n
	Generates nGram Model.
    '''
    vocabulary = set(MergedText)

    MergedText = delete_short(MergedText,ngramModel) # delete the words from short sentence where words amount is less than n

    nngrams = boundaries(list(ngrams(MergedText,ngramModel)))  #generate the nngrams without cross boundary

    cfd = ConditionalFreqDist()
    ngramSet = set()
    vocabularyOfWords = set()
    fdist = FreqDist()

    ProbDictionary = defaultdict(list)

	#Generate conditional frequency distribution
    for ngram in nngrams:
        ngramSet.add(ngram)
        initial_text = tuple(ngram[:-1]) #this is the initial_text from the ngram (n-1)
        last_word = ngram[-1] #this is the last word from ngram
        cfd[initial_text][last_word] += 1

        #Smoothing and generating probabilities using Laplace Algorithm
        #laplace_prob = [1.0 * (1+cfd[initial_text][last_word]) / (len(vocabulary)+cfd[initial_text].N())]
        ProbDictionary[initial_text].append(last_word) #Storing probability of each word
        vocabularyOfWords.add(last_word)

    proDic={} #this is the probabilty stored for each words after each ngrams
    for key,value in ProbDictionary.items():
        words=set(value)
        proDic[key]={}
        sum_freq=0
        for word in words:
            fre_word=value.count(word)/float(len(word)) #compute the count of each words for a gram
            proDic[key][word]=fre_word
            sum_freq+=fre_word
        for key_,value_ in proDic[key].items():
            proDic[key][key_]=value_/sum_freq  #normalization of the propability
    #generate sentences
    generateSentences(nngrams,cfd,ngramModel,proDic,numSentences)

def find_start_grams(nngrams):
    '''this function finds all the grams that can be used at the beginning of a sentence, which has a start tag'''
    start_grams=[]
    for gram in nngrams:
          if gram[0]=='start_':
             start_grams.append(gram)
    return start_grams

def generateSentences(nngrams,cfd,ngramModel,proDic,numSentences):

    for sentence in range(int(numSentences)):

        #select a random ngram out of all ngrams generated
        start_gram=find_start_grams(nngrams) #find the grams that follows the start mark
        random_ngram = random.choice(start_gram)

        #get rid of the last word from the ngram, "seed" is a list of string with the first (n-1) words
        seed = tuple(random_ngram[:-1])

        #Predict the next word based on the seed
        value=proDic[seed]
        seed_=random.uniform(0,1) #random generate a float number
        calculate_fre=0
        for word,freq in value.items():
            calculate_fre+=freq
            if seed_<calculate_fre:
                predictedWord = word #find the predicted word based on the distribution
                break


        #New text generated by joining seed and the predicted word
        newText = ' '.join(seed) + " " + predictedWord
        end_marks=['.','?','!']

        #Predict next words for new sentence creation
        while (predictedWord!='end_'):  # keep going until find an end mark
            #Add the word predicted to the actual seed
            seed+= tuple([predictedWord])

            #Convert the seed to a list
            seedList = list(seed)

            #Remove the first element from the seedList to create a new seed...slides to the next seed
            seedList.pop(0)

            #Convert the updated seed back to the seed tuple
            seed = tuple(seedList)

            #Predict the next word based on the new seed
            for key,value in proDic.items():
              if(key == seed):
               seed_=random.uniform(0,1)
               calculate_fre=0
               for word,freq in value.items():
                    calculate_fre+=freq
                    if seed_<calculate_fre:
                        predictedWord = word
                        break


            #New Text is created by joining previous newText and the next predicted word
            if predictedWord!='end_':
               newText = newText + " " + predictedWord
        print("New Sentence: ", newText[7:])

def main():
    '''
    This is the main function.
	'''
    start=time.time() #record the start time

    #Fetch arguments in variables
    ngramModel= int(sys.argv[1]) #n=1,2,3,4 and so on for the number of ngram model to be generated
    numSentences = sys.argv[2] #number of sentences to be generated
    numFiles = len(sys.argv)   #number of text files provided in argument

    #This function reads the files
    MergedText = read_files(ngramModel,numFiles)

    print("This program generates random sentences based on an Ngram model.")
    print("Command line settings: ngram.py",ngramModel,numSentences)

    #this function generates the ngram model and random sentences
    generateModel(MergedText,ngramModel,numSentences)

    #record the end time
    end = time.time()
    total_time = end-start

    filename=""
    for file in range(3,numFiles):
        #Read all the files
        filename += sys.argv[file] + " "

    #Write the log file
    with open("ngram-log.txt", "a+") as text_file:
        text_file.write("%s" % total_time + " secs python ngram.py " + "%s %s %s" % (ngramModel, numSentences, filename) + "\n")

if __name__ == '__main__':
    main()
