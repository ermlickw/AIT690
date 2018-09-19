'''
AIT 690 | Assignment 1 | Due 9/19/2018
Billy Ermlick
Nidhi Mehrotra
Xiaojie Guo
************************************************************************************************************************************************
This code aims to implement a dialogue robot Eliza who can engage in a dialogue with the user.
Eliza will begin the dialogue by asking the name of the user. Current implementation of Eliza contains:

1)	"Word Spotting" function to extract the keywords from users' answer and give related feedback.
    (e.g. 'sad' to 'What makes you sad?')
2)	 Transforming Sentences from user to the answers/questions from Eliza.(e.g " I am not sure what to do." to "You are not sure what to do?")
3)	Personalized the conversation by using user name in some of the questions
4)	Robust answers in some plausible way when the users enters some gibberish sentence or the sentences are not understandable
    (e.g.I DIDN'T QUITE UNDERSTAND. CAN YOU SAY THAT ANOTHER WAY?)
5)	Used lambda function for asking default questions to the user, such as storing and returning the user's favorite things throughout discourse.

***********************HOW TO RUN THE PROGRAM:****************************************************************************************************
When using the code, run the Assignment1.py, type your response into the prompt and press ENTER. Do not use contractions in your reply.

***********************HOW TO END THE PROGRAM:****************************************************************************************************
Program can be ended by using any of these keywords
•	bye
•	farewell
•	adios
•	see you later
•	talk to you later

************************************************************************************
SIMPLE EXAMPLE DIALOGUE:
************************************************************************************
Simple Example Dialogue:
HELLO, MY NAME IS ELIZA. I AM A PSYCHOTHERAPIST. WHAT IS YOUR NAME?
    'My name is Coco'
NICE TO MEET YOU, COCO. HOW DO YOU FEEL TODAY?
    'i fee l sAD...'
"I DIDN'T QUITE UNDERSTAND. CAN YOU SAY THAT ANOTHER WAY?"
    'I feel sad'
TELL ME MORE.
    'I always have lot of work to do'
CAN YOU THINK OF A SPECIFIC EXAMPLE?
    'Yes, I need to read 10 paper a week'
ARE YOU SURE?
    'Pretty sure'
COCO, WHY PRETTY SURE?
    'can you sing a song for me?'
Sure
    'I want to share my favorites'
COCO, WHY DO YOU WANT TO SHARE YOUR FAVORITES?
    'favorite is an interesting topic'
SPEAKING OF FAVORITES... I'D LIKE TO LEARN MORE ABOUT YOU
I know your favorite song is ...well maybe I am not sure....
...What is your favorite song?
    'My favorite song is Raise you up'
GOT IT. WHAT ELSE IS ON YOUR MIND?
    'Nothing! bye"
I HOPE THIS CONVERSATION WAS PRODUCTIVE. GOODBYE.
***************************************************************************************

More detailed information is stated below.
'''

import re
import random
import string
from collections import defaultdict
from string import punctuation
import nltk
from nltk.corpus import words
from nltk.corpus import wordnet
from string import digits
from nltk.stem import WordNetLemmatizer

eliza = (" \
                ||||||||||||||||||||||||||||||||||||||||||||||||| \n \
                ************************************************* \n \
                ||||||||||   ||            ||||||||        |||    \n \
                ||           ||      ||         |||       || ||   \n \
                |||||||      ||               |||        ||   ||  \n \
                ||           ||      ||     |||         ||||||||| \n \
                ||||||||||   ||||||  ||   ||||||||     ||      || \n \
                ************************************************* \n \
                ||||||||||||||||||||||||||||||||||||||||||||||||| \n \
               \n")

#these global variables are used to store and retrieve the user's favorite things
defaultReply = "...well maybe I am not sure."
myfavorites = defaultdict(lambda: defaultReply)
index = ['animal', 'color', 'flavor', 'holiday']

def extract_username(userName):
    '''
    This function introduces Eliza to the User, asks the User their name and begins the conversation.
    '''
    userName = input('Hello, my name is Eliza. I am a psychotherapist. What is your name? \n'.upper()).strip().lower().strip(punctuation)
    userName,userNameIsGood = check_valid_user_name(userName) #true or false

    while not(userNameIsGood): #until we have a good user name
        userName = input("I'm sorry I didn't catch that. What is your name again? \n").strip().lower().strip(punctuation).upper()
        userName, userNameIsGood = check_valid_user_name(userName)


    print(('Nice to meet you, ' + userName + ". ").upper())
    return userName


def check_valid_user_name(userName):
    '''
    This function extracts the username from the response and checks if the
    user's inputted name is valid.
    '''
    if name_retrieval('is\s(.*)',userName):  # get the word after "is"
        name=name_retrieval('is\s(.*)',userName)
    elif name_retrieval('am\s(.*)',userName):   # get the word after "am"
        name=name_retrieval('am\s(.*)',userName)
    else:  # if neither of those words, just get the text
        name=userName.partition(' ')[0]
    #error checking:
    if len(name) >=10:
        return [], False
    return name, True

def name_retrieval(re_pattern,content):
    '''
    This function returns either the name of the user or false, indicating that the
    result is not present.
    '''
    query=re.compile(re_pattern,re.MULTILINE|re.DOTALL)
    if len(re.findall(query,content))>0:
        return re.findall(query,content)[0]
    else:
        return False

def checkgibberish_words(userInput):
    '''
    This function checks if all the words of the input sentence is a valid English word.
    '''
    #Remove punctuation

    #make translator object
    translator=str.maketrans('','',string.punctuation)
    userInput=userInput.translate(translator)

    #Remove digits from the input
    userInput_NoDigits = re.sub(r'\d+', '', userInput)
    wnl = WordNetLemmatizer()

    gibberishWord = False
    for word in userInput_NoDigits.split():
        chkWord=word in words.words()
        if not chkWord:
                chkWordNet = word in wordnet.words()
                if not chkWordNet:
                     lemma = wnl.lemmatize(word, 'n')
                     #WordNet and Word dictionary does not include plural words,
                     #hence checking for plural words through WordNetLemmatizer
                     plural = True if word is not lemma else False
                     if not plural:
                        gibberishWord = True

    return gibberishWord

def determine_reply(userInput, userName):
    '''
    This function analyzes the users response and determines what Eliza
    should say in response. The function takes the user's response, filters
    the text through various analyzers and determines what response should be
    provided to the user. Bulk of program is provided here.
    '''
    repetitionList =['Can you repeat that, ' + userName.upper() + '? \n', "I'm not sure I follow? \n",
                     "I didn't quite understand, can you say that another way? \n",]

    def transform(Input):
        '''
        This function is used to convert the user's sentence into a question if
        no particular words are identified by the if statements below. This is
        done to provide a robust reply for non-preprogramed inputs.
        '''
		#Replace "i" with "you"
        output = re.sub(r'\bi\b',r'-1-',Input)

		#Replace "am" with "are"
        output = re.sub(r'\bam\b',r'-2-',output)

		#Replace "my" with "your"
        output = re.sub(r"\bmy\b",r'-3-',output)

		#Replace "was" with "were"
        output = re.sub(r"\bwas\b",r'-4-',output)

		#Replace "me" with "you"
        output = re.sub(r"\bme\b",r'-5-',output)

		#Replace "mine" with "yours"
        output = re.sub(r"\bmine\b",r'-6-',output)

		#Replace "you" with "I"
        output = re.sub(r"\byou\b",r'-7-',output)

		#Replace "your" with "my"
        output = re.sub(r"\byour\b",r'-8-',output)

		#Replace "yours" with "mine"
        output = re.sub(r"\byours\b",r'-9-',output)

		#Replace "are" with "am"
        output = re.sub(r"\bare\b",r'-10-',output)

		#Replace "were" with "was"
        output = re.sub(r"\bwere\b",r'-11-',output)

        #Replace "it has" with "has it"
        output = re.sub(r"\bit has\b",r'-12-',output)

        output = re.sub(r'-1-',r'you',output)
        output = re.sub(r'-2-',r'are',output)
        output = re.sub(r"-3-",r'your',output)
        output = re.sub(r"-4-",r'were',output)
        output = re.sub(r"-5-",r'you',output)
        output = re.sub(r"-6-",r'yours',output)
        output = re.sub(r"-7-",r'I',output)
        output = re.sub(r"-8-",r'my',output)
        output = re.sub(r"-9-",r'mine',output)
        output = re.sub(r"-10-",r'am',output)
        output = re.sub(r"-11-",r'was',output)
        output = re.sub(r"-12-",r'has it',output)

        return output

    #Check if the sentence has any gibberish word / typos
    gibberishWord = checkgibberish_words(userInput)
    if gibberishWord:
        output = "I didn't quite understand. Can you say that another way?"
        return output.upper() + '\n', True

    #If block to search for inputs starting with "how"
    #Done to answer the user if they ask a how type question.
    #This is done for other similar words as shown below.
    if re.search(r"^(how) (.*)",userInput):
        #Reply is selected from this list
        howRepliesList = ['What do you think?', 'Why do you ask?',]
        output = random.choice(howRepliesList)
        return output.upper() + '\n', True

    if re.search(r"^(what) (.*)",userInput):
        #Reply is selected from this list
        whatRepliesList = ['What do you think?', 'Why do you ask?',]
        output = random.choice(whatRepliesList)
        return output.upper() + '\n', True

    if re.search(r"^yes*\B",userInput):
        #Reply is selected from this list
        yesRepliesList = ['I see...', 'Are you sure?',]
        output = random.choice(yesRepliesList)
        return output.upper() + '\n', True

    if re.search(r"^no*\B",userInput):
        #Reply is selected from this list
        noRepliesList = ['why are you being so negative', 'Are you always this pessimistic?',]
        output = random.choice(noRepliesList)
        return output.upper() + '\n', True

    if re.search(r"\bbecause.*",userInput):
        #Reply is selected from this list
        becauseRepliesList = ['Is that the real reason?',]
        output = random.choice(becauseRepliesList)
        return output.upper() + '\n', True

    #If block to search for inputs starting with "Can you"
    if re.search(r"^can you.*",userInput):
        #Partition the input to find the context of it
        before, mid, after = userInput.partition('can you')
        #Reply is selected from this list
        canRepliesList = ['Sure','What made you think I can' + transform(after) + '?', 'Dont you believe I can' + transform(after) + '?']
        output = random.choice(canRepliesList)
        return output + '\n', True

    #If block to search for inputs starting with "Can I"
    if re.search(r"^can i.*",userInput):
        #Partition the input to find the context of it
        before, mid, after = userInput.partition('can i')
        #Reply is selected from this list
        canRepliesList = ['Sure','What makes you think you can' + transform(after) + '?']
        output = random.choice(canRepliesList)
        return output + '\n', True

    #If block to search for inputs starting with "Who"
    if re.search(r"^who.*",userInput):
        return "Who do you think?".upper() + '\n', True

    #If block to search for inputs starting with "Hi or Hello"
    if re.search(r"\b(hi|hello)\b",userInput):
        return "Hello again. What's on your mind?".upper() + '\n', True

    #If block to search for inputs having "all" in the sentence
    #this is to reply to instances where the user generallizes something
    if re.search(r".* all .*",userInput):
        return "In what way?".upper() + '\n', True

    #If block to search for inputs having "always" in the sentence
    # this is done to reply to instances where the user generalizes something
    if re.search(r".* always .*",userInput):
        return "Can you think of a specific example?".upper()+ '\n', True

    #If block to search for inputs having the keyword "depressed/sad/upset" and sending reply based on it
    #this is doneto reply to the user when they express specific feelings
    if re.search(r"\b(depressed|sad|upset|unhappy|angry|positive|optimistic|fearful|happy)\b",userInput):
        output = re.sub(r".*\b(depressed|sad|upset|unhappy|angry|positive|optimistic|fearful|happy)\b.*",
               r"What made you \1? \n",userInput)
        return output.upper(), True

    if re.search(r"\bfeel*\B",userInput):
        #Reply is selected from this list
        feelRepliesList = ['Tell me more...',]
        output = random.choice(feelRepliesList)
        return output.upper() + '\n', True


	#If block to search for inputs having the keyword "suffering" and sending reply based on it
    if re.search(r"\bsuffering\b",userInput):
        return "How can I help you?".upper()+ '\n', True

    #If block to search for inputs having "gave me" in the sentence
    if re.search(r"\b(gave) me\b",userInput):
        output = re.sub(r".*\b(gave)\b.*",
               r"What made them give you that?",userInput)
        return output.upper(), True

    #If block to search for inputs having "computer" in the sentence
    if re.search(r".*computer.*",userInput):
        return "Are you talking about me?".upper() + '\n', True

    #If block to search for inputs having the keyword "favorite" and store this information for later use
    if re.search(r"\bfavorite\b",userInput):
        print("speaking of favorites... I'd like to learn more about you".upper())
        cat = random.choice(index)
        diffcat = cat
        i=1
        print("I know your favorite " + cat + " is " + myfavorites[cat] + "...")
        while myfavorites[diffcat] != defaultReply and i < 30:
            diffcat = random.choice(index)
            i=i+1
        if i <30:
            myfavorites[diffcat] = input("...What is your favorite " + diffcat + "? ")
            output = "Got it. What else is on your mind?"
        else:
            output = "I guess I know everything about you..."
        return output.upper(), True


	#Good bye list
    if re.search(r"\b(bye|farewell|adios|see you later| talk to you later)\b",userInput):
        return "\n", False

    # if there is no match ask them a question form of their own input. This is
    #Done to provide a robust reply to various inputs.
    else:

       transformed_text = transform(userInput).upper()

       #Created lambda function for asking default questions to the user
       defaultRepliesList = ['Why ' + transformed_text ,transformed_text, 'Why do you think so']
       defaultoutput = userName + ', ' + random.choice(defaultRepliesList)
       mylist = defaultdict(lambda: defaultoutput)
       mylist['I AM'] = 'Why are you'
       mylist['YOU ARE'] = 'Why do you think I am'
       mylist['I'] = 'Why do you'
       mylist['I DO'] = 'Why do you'

       if re.search(r".*i am.*",userInput):
            append_word = mylist['I AM']
            before, mid, after = transformed_text.partition('YOU ARE')
            output =  userName + ', ' + append_word + after
       elif re.search(r".*i do.*",userInput):
            append_word = mylist['I DO']
            before, mid, after = transformed_text.partition('DO')
            output =  userName + ', ' + append_word + after
       elif re.search(r".*you are.*",userInput):
            append_word = mylist['YOU ARE']
            before, mid, after = transformed_text.partition('I AM')
            output =  userName + ', ' + append_word + after
       elif re.search(r"^\bi\b.*",userInput):
            append_word = mylist['I']
            before, mid, after = transformed_text.partition('YOU')
            output =  userName + ', ' + append_word + after
       else:
            defaultreply = mylist['default'] #default response
            output =  defaultreply

       return output.upper() + '? \n', True



def main():
    '''
    This is the main function. Conversation is started.
    conversation is continued until an ending is reached. The conversation is ended.
    '''
    #Variable Initialization
    converse = True
    userName = ""

	#Eliza will introduce from one of the replies from this introduction list
    introductionList =['What is on your mind today? \n', 'How do you feel today? \n',]

	#Eliza will end the conversation with his statement
    goodbyeList =['I hope this conversation was productive. Goodbye. \n','Goodbye. \n', 'Farewell \n',]

    print(eliza)

    #main function:
	# start conversation and get user's name
    userName = extract_username(userName)

	#initiate conversation dialogue, choose from introduction list
    userInput = input(random.choice(introductionList).upper()).strip(punctuation).lower()

	#while conversation continues:
    while converse:
        reply, converse = determine_reply(userInput, userName) #determine a reply based on user input and if conversation should continue
        if converse:
            userInput = input(reply).strip().lower().strip(punctuation) #if there is a reply allow user to respond

	#say goodbye if user types bye or farewell or adios
    print(random.choice(goodbyeList).upper())


if __name__ == '__main__':
    main()
