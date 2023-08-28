# Content Moderator Discord Bot

O-----------------------------------------------------------------------------------------------------------------------------------O
O-----------------------------------------------------------------------------------------------------------------------------------O


## Instructions for Use

This project intends to create a discord bot that is able to autonomously moderate a server using natural
language processing. 

To run the bot on your local machine, clone the repository. Then navigate to Content-Mod-DiscordBot/bots/run.py
Running this script (using your own discord bot token) will download the model files needed to run the bot and start 
it up on your server.

#### Current Functionality as of 8/28/23:
- Censoring of generally innapropriate/toxic messages
- Basic punitive actions against users who repeatedly send such messages (warning, muting, blocking)
*- Some summarization of message content on flagged messages. There is an asterisk in front of this
  "functionarlity" because it is a placeholder for a more robust model intended to replace it. However,
  it is part of the bot though.

#### Features Under Development as of 8/28/23:
- Named Entity Recognition model to apply to flagged messages intened to identify which users are being or
  groups are being targeted within innapropriate messages
- !appeal feature for false positives including more sophisticated model for reconsidering such messages

#### Features Currently on Hold from Development as of 8/28/23:
- summarization model to provide insight into why messages were censored


O-----------------------------------------------------------------------------------------------------------------------------------O
O-----------------------------------------------------------------------------------------------------------------------------------O

## Current State of Project --- (ReadMe last updated on 8/28/23)
The functionality of the bot remains the same. Current effort has been spent on R&D for a model that will provide reasons for why 
flagged messages were flagged by the Rule Adherance Classifier Model. 

Initially, I explored fine tuning T5 for Conditional Generation with an augmented version of the wikipedia toxic comments dataset. 
This effort failed. I believe the main reasons for this being that the augmentation process I used produced poor quality data. 
Secondly, I believe there may be something structurall stopping me from using the exact method/implementation of T5 for this objective
of providing reasons as to why a message is innapropriate. Prior to tokenization, I have been appening "summarize: " to examples. This 
is standard and expected for summarization tasks, but I began to question whether providing a reason why a message is innapropriate 
shares enough in common with the task of summarization to implement the fine tuning in this way. I am thinking about and looking into 
this. For now though, this model holds less precedence than the NER model which I will describe in the next paragraph.

Currently, I am building a model that will perform Named Entity Recognition on discord messages. With the goal being to identify not 
just standard named entities like Organization, Person, Location, but to also tag Inapropriate entities. To accomplish this I am going 
to build out an augmented version of the wikipedia toxic comments dataset using both a more standard NER model and the Rule Adherance
Classifier Model (Each of which I have alreadyfine tuned for this task). A more detailed description of this plan can be found within 
the model_training_docs/NER_model/NER_model_building_pipeline.txt. 


O-----------------------------------------------------------------------------------------------------------------------------------O
O-----------------------------------------------------------------------------------------------------------------------------------O

#### 8/14/23 Update

The rule adherence classifier has been implemented into the bot. Currently there is basic content moderation
functionality. This means that innapropriate messages are hidden by a discord spoiler when detected. There is 
some minor punitive action features for when innapropriate messages persist from a single user repeditively.
However, error handling needs to be implemented. There is some bug that is causing error in that process.

I have also created a stub for an !appeal command for when the model incorrectly predicts an innapropriate 
message. My plan here is to send messages for appeal to a more sophisticated rule adherence classifier model
that will be better able to label edge cases. Thus saving on computational requirements when running on busy
servers.

Currently, model files are stored in the following google drive link 

https://drive.google.com/drive/folders/1MUpmOU9G1g0DljfV35ddwehzG0w1S69W?usp=sharing

I have added a local_models directory, which currently is empty except for an instructions.txt file that contains
directions for how to download and store models. At this writing, this downloading process is manual. I intend to 
automate the downloading of these files soon however. Once the Models folder from that drive link are correctly 
placed in the local_models dir, running bot.py with your own bot token should work on any local machine. 

At this stage of development, I am working on expanding the functionality of the bot by training a summarization 
model that will be used to explain to users who have made innapropriate comments why their comments were innapropriate.

I have added a diagram of features I intend to add to the bot in the systems_diagram directory. There is also a .txt 
explanation of the diagram there.

O-----------------------------------------------------------------------------------------------------------------------------------O
O-----------------------------------------------------------------------------------------------------------------------------------O

#### 8/14/23 Update

The rule adherance classifier for basic innapropriate/toxic behavior has been trained. The
model is a distilbert uncased fine tuned on the wikipedia toxic comment dataset for binary 
sentiment analysis. The next step is to implement this model into the discord bot.

O-----------------------------------------------------------------------------------------------------------------------------------O
O-----------------------------------------------------------------------------------------------------------------------------------O

#### 8/9/23 Update

Currently, the project is in its infancy. I am building out the functionality of the bot as well as training the 
binary sentence classifier. 

O-----------------------------------------------------------------------------------------------------------------------------------O
O-----------------------------------------------------------------------------------------------------------------------------------O

If you want to help out, submit a pull request with features you feel are valuable. If you can successfully trained
the summary model as specified in its docs that would be most appreciated.

If it checks out when tested, it will be committed :)


