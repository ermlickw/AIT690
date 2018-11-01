# AIT 690 | Assignment 1 | Due 9/19/2018
### Billy Ermlick
### Xiaojie Guo
### Nidhi Mehrotra
This code aims to implement a dialogue robot Eliza who can engage in a dialogue with the user.
Eliza will begin the dialogue by asking the name of the user. Current implementation of Eliza contains:
1)	"Word Spotting" function to extract the keywords from users' answer and give related feedback (e.g. 'sad' to 'What makes you sad?')
2)	 Transforming Sentences from user to the answers/questions from Eliza (e.g., " I am unsure on what to do." to "You are unsure on what to do?")
3)	Personalized the conversation by using user name in some of the questions
4)	Robust answers in some plausible way when the users enters some gibberish sentence or the sentences are not understandable (e.g., I'm sorry I didn't catch that. What is your name again?)
5)	Used lambda function for asking default questions to the user, such as storing and returning the user's favorite things throughout discourse. <br><br>
HOW TO RUN THE PROGRAM:<br>
When using the code, just run the Assignment1.py and type your response into the prompt and press ENTER. Do not use contractions.<br>
***********************************************************************************
Simple Example Dialogue: <br>
HELLO, MY NAME IS ELIZA. I AM A PSYCHOTHERAPIST. WHAT IS YOUR NAME?<br>
    'My name is Coco'<br>
NICE TO MEET YOU, COCO. HOW DO YOU FEEL TODAY? <br>
    'i fee l sAD...'    <br>
"I DIDN'T QUITE UNDERSTAND. CAN YOU SAY THAT ANOTHER WAY?"<br>
    'I feel sad'<br>
TELL ME MORE.<br>
    'I always have lot of work to do'<br>
CAN YOU THINK OF A SPECIFIC EXAMPLE?<br>
    'Yes, I need to read 10 paper a week'<br>
ARE YOU SURE?<br>
    'Pretty sure'<br>
COCO, WHY PRETTY SURE?     <br>
    'can you sing a song for me?'<br>
Sure<br>
    'I want to share my favorites'<br>
COCO, WHY DO YOU WANT TO SHARE YOUR FAVORITES?    <br>
    'You should know my favorite things'<br>
SPEAKING OF FAVORITES... I'D LIKE TO LEARN MORE ABOUT YOU<br>
I know your favorite song is ...well maybe I am not sure....<br>
...What is your favorite song?  <br>
    'My favorite song is Raise you up'   <br>
GOT IT. WHAT ELSE IS ON YOUR MIND?     <br>
    'Nothing! bye"<br>
I HOPE THIS CONVERSATION WAS PRODUCTIVE. GOODBYE.<br>
***************************************************************************************<br>
# AIT690-Assignment2

This Python program called ngram.py will learn an N-gram language model from an arbitrary number of plain text files. The program can generate a given number of sentences based on that N-gram model.

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

**************************************************************************************<br>
# AIT690-Assignment3

This is a python program which assigns parts of speech tags to a training file
which maximize P(tag|word). For words which are not included in the training file,
they are assumed to be NNself. Words which only have one part of speech in the training
data are labeled as that part of speech in the test file. Words with multiple
potential parts of speech which have unlabeled neighbors are tagged as their
most likely tag in the training dataset. After this proceedure, untagged words
with tagged neighbors were assigned based on maximizing their conditional
probabiities. The accuracy of our model before additional POS rules were applied
was %55.17. After the addition of the rules, our accuracy increased to 80.87%.

The labeled training data is "pos-train.txt"
The untagged test file is "pos-test.txt"
The predicted labeled test data is "pos-test-with-tags.txt"
The golden standard labeled test data is "pos-test-key.txt"
The scoring file is "scorer.py"

"pos-tagging-report.txt" and "tagger-log.txt" are  reporting and logging files,respectively.


The script can be run by entering: <br>
$  python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt <br>
$ python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-taggingreport.txt<br>

Some of the code for the probability tables and confusion matrix was obtained from the NTLK Book.
https://www.nltk.org/book/

Some of the rules were obtained from the Speech and Language Processing Book by Jurafsky et al.

**************************************************************************************<br>
# AIT690-Assignment4

This is a python program which assigns parts of speech tags to a training file
which maximize P(tag|word). For words which are not included in the training file,
they are assumed to be NNself. Words which only have one part of speech in the training
data are labeled as that part of speech in the test file. Words with multiple
potential parts of speech which have unlabeled neighbors are tagged as their
most likely tag in the training dataset. After this proceedure, untagged words
with tagged neighbors were assigned based on maximizing their conditional
probabiities. The accuracy of our model before additional POS rules were applied
was %55.17. After the addition of the rules, our accuracy increased to 80.87%.

The labeled training data is "pos-train.txt"
The untagged test file is "pos-test.txt"
The predicted labeled test data is "pos-test-with-tags.txt"
The golden standard labeled test data is "pos-test-key.txt"
The scoring file is "scorer.py"

"pos-tagging-report.txt" and "tagger-log.txt" are  reporting and logging files,respectively.


The script can be run by entering: <br>
$  python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt <br>
$ python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-taggingreport.txt<br>

Some of the code for the probability tables and confusion matrix was obtained from the NTLK Book.
https://www.nltk.org/book/

Some of the rules were obtained from the Speech and Language Processing Book by Jurafsky et al.

**************************************************************************************<br>

PROJECT

DATA available via -> 
