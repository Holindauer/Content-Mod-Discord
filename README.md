# Content Moderator Discord Bot

### Vision for Project
The goal for this repository is to have create a discord bot that utilizes natural language processing
to monitor the adherance of discord server rules. The current vsion for this project is as follows:

A discord bot will monitor all incoming messages into a given server. These messages will be passed through
a binary autoencoding transformer trained to perform sentence classification, predicting whether a given
sentance follows the rules of the server. In the instance the classifier predicting the rules have been broken, 
the flagged message will be sent to two additional nlp models. 

The first of these models will be an anutoregressive
language model fine tuned to generate an explanation of why the rules have been broken. This model has is 
given a further stipulation that its explanation must also follow the rules of the server. I believe this
stipulation may require using the sentence classifier within its loss to impose a penalty if that explanation
be innapropriate as well. The other model of which the flagged sentence is sent to will be fine tuned to perform 
Named Entity Recognition of those words in that were deemed innapropriate. 

These explanations and flagged words will be sent to the rule breaking user in dms. This aspect of the bot is
less thought out at current, but I imagine it could look something like: Each flagged message warrants a warning
either in dm or the channel it was sent. Then if 5 innapropriat emessages in a row implement a timeout or a 
ban. Something of that nature.

### Current State of Project --- (ReadMe last updated on 8/14/23)

The rule adherance classifier for basic innapropriate/toxic behavior has been trained. The
model is a distilbert uncased fine tuned on the wikipedia toxic comment dataset for binary 
sentiment analysis. The next step is to implement this model into the discord bot.


#### 8/9/23 update

Currently, the project is in its infancy. I am building out the functionality of the bot as well as training the 
binary sentence classifier. 

If you want to help out, submit a pull request. If it checks out, it will be committed. 


