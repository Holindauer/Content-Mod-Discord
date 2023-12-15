import discord
from discord.ext import commands
import datetime
import asyncio


class Bot():
    def __init__(self, local_RAC, token):
        self.dash_line = "-"*50 + "\n"
        print(f"{self.dash_line*3}\nInsantiating Content Moderator Bot...")

        self.token = token
        self.local_RAC = local_RAC

    def start(self):
        print("Starting Content Moderator...")

        #set intents
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True  
        intents.guilds = True
        intents.members = True

        #instantiate bot with the above intents
        bot = commands.Bot(command_prefix='!', intents=intents) 

        #when bot goes live notify all channels
        @bot.event
        async def on_ready():
            print('We have logged in as {0.user}'.format(bot))
            for guild in bot.guilds:
                for channel in guild.text_channels:
                    await channel.send('Content Moderator is Live...')


        #dictionary to keep track of users and their number of rule violations
        user_violations = {}

        #check for incoming messages and apply moderation --- This is the core of the bot
        @bot.event
        async def on_message(message):

            # return if the message author is the bot
            if message.author == bot.user:
                return
            
            # inference on incoming messages           
            if self.local_RAC.run(message.content):

                #censor innapropriate message
                await message.delete() 

                # Notify the user of the violation
                censored_msg = f'**Message by {message.author.mention} is inappropriate and was censored: ||{message.content}||**'
                await message.channel.send(censored_msg, delete_after=7)  
                await message.channel.send('Channels must be used for official purposes only!', delete_after=7)


            await bot.process_commands(message)


        bot.run(self.token)




