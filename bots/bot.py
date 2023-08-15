import discord
from discord.ext import commands

from run_local_RAC import RAC  #local rule adherance classifier



def main():
    #instantiate local rule adherance classifier
    local_RAC = RAC()

    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True  # This one is important!
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
    @bot.event
    async def on_message(message):
        
        print(type(message.content))
        print(str(message.content))


        # return if the message author is the bot
        if message.author == bot.user:
            return
        
        rule_adherance = local_RAC.run(message.content) #either 1 or 0

        #inference on incoming messages
        if rule_adherance:
            await message.channel.send("This message is inappropriate!")
        else:
            await message.channel.send("This message is appropriate!")

        await bot.process_commands(message)



    bot.run(token)




token =  "Token Here"

main()