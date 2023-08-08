import discord
from discord.ext import commands
import pandas as pd

'''
This bot will be used to text the content moderation capabilities of
the content moderator bot. It will send messages from the wikipedia
toxic comments dataset that should trigger the content moderator to
flag them. 

'''

def main():
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.members = True
    bot = commands.Bot(command_prefix='!', intents=intents)


    #when bot goes live notify all channels
    @bot.event
    async def on_ready():
        print('We have logged in as {0.user}'.format(bot))
        for guild in bot.guilds:
            for channel in guild.text_channels:
                await channel.send('Tester Bot is Live...')


    #load toxic comments 
    df = pd.read_csv("C:\\Users\\hunte\\OneDrive\\Documents\\Coding Projects\\Content-Moderator\\bots\\tester_bot\\toxic_df.csv")
    print("Dataframe loaded successfully.\n")



    #send toxic comments to all channels for testing
    @bot.command()
    async def send_tox(ctx, num_samples: int):
        print(f"Starting !send_tox with {num_samples} samples...")  # Printout to terminal

        if num_samples <= 0:
            await ctx.send("Please provide a positive number of samples.")
            return

        # Make sure num_samples doesn't exceed the length of the DataFrame
        num_samples = min(num_samples, len(df))

        # Sample 'num_samples' random rows from the DataFrame
        random_rows = df.sample(n=num_samples)
        
        # Loop over all text channels in the guild
        for guild in bot.guilds:
            for channel in guild.text_channels:
                for index, row in random_rows.iterrows():
                    await ctx.send(f"Sending {num_samples} toxic comments to {channel}...")
                    await channel.send(row['comment_text'])

                
    bot.run(token)




token = #Token goes here

main()