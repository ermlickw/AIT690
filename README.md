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
