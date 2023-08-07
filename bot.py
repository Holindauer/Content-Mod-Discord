import discord
from discord.ext import commands

'''
Warren:

I'll briefly explain what the bot is going to be doing and what to add to the script.
The bot is going to be scanning all incoming messages and checking if any of them break
the rules of the server. We'll prototype in python to quickly build up the functionality 
we want. Then eventually, we'll switch to a language that is faster with more parralelism.

My plan for the bot is to collect every new message in the server and run them through a 
neural network that will predict if the message is breaking the rules. If the message is
breaking the rules then it will be sent to two other neural networks. One of the networks
will generate a summary of why the message broke the rules. The second network will identify
specific words that break the rules. There will also be some functionality for when the rules
have been broken too many times. This could look like a few things. I was thinking something
like a timeout or a ban.

I am currently working on the classifier model that will predict if the message is breaking
the rules. 

In the meantime, would you implement a mechanism that will collect all the messages in order to
be run through the models. Have the messages save to a csv file for later use. Just create a function stub for 
the calls to the model. Once the models are ready well replace the stubs. Also, create a mechanism 
that counts how many times the rules have been broken and if they've exceeded some threshold, then carry 
out the user timeout. Test it out in a server and update the repository with your implementation.

'''



def main():
    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    intents.members = True

    bot = commands.Bot(command_prefix='!', intents=intents)


    #when bot goes live notify all channels
    @bot.event
    async def on_ready():
        print('We have logged in as {0.user}'.format(bot))
        for guild in bot.guilds:
            for channel in guild.text_channels:
                await channel.send('Content Moderator is Live...')


    #check for incoming messages
    messages_reciece = []
    @bot.event
    async def on_message(message):
        
        # return if the message author is the bot
        if message.author == bot.user:
            return
        
        # Send a message to the channel the original message came from
        await message.channel.send("I received a message!")

        messages_reciece.append(message) 

        await bot.process_commands(message)



    bot.run(token)




token =  #Token goes here

main()