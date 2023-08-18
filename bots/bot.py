import discord
from discord.ext import commands
import datetime
import asyncio


class Bot():
    def __init__(self, local_RAC, token):
        dash_line = "-"*50 + "\n"
        print(f"{dash_line*3}\nInsantiating Content Moderator Bot...")

        self.token = token
        self.local_RAC = local_RAC

    def start(self):
        print("Starting Content Moderator...")

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


        #dictionary to keep track of users and their num rule violations
        user_violations = {}


        #check for incoming messages
        @bot.event
        async def on_message(message):
            
            #threshold for action to be taken
            threshold_warning = 5
            threshold_timeout = 10
            threshold_ban = 15

            # return if the message author is the bot
            if message.author == bot.user:
                return
            
            #verify rule adherance
            rule_adherance = self.local_RAC.run(message.content) #bianry output

            #inference on incoming messages
            if rule_adherance:
                user_id,username = message.author.id, message.author.name #get username and id
                user_violations[user_id] = user_violations.get(user_id, 0) + 1 #increment user violations
                print(f'User {username} has {user_violations[user_id]} violations.')


                #censor innapropriate message
                await message.delete()  # Delete the offending message
                censored_msg = f'**Message by {message.author.mention} is inappropriate and was censored: ||{message.content}||**'
                await message.channel.send(censored_msg)
                await message.channel.send('If you wish to appeal this decision, please type !appeal in the chat.')

                #ban user
                if user_violations[user_id] == threshold_ban:
                    await message.author.send(f"@{username}, You have been banned for sending innapropriate messages!")
                    await message.author.ban(reason="Sending innapropriate messages")

                #timout user
                elif user_violations[user_id] >= threshold_timeout:
                    mute_role = discord.utils.get(message.guild.roles, name="Muted")

                    if not mute_role:
                        # Create the mute role if it doesn't exist
                        mute_role = await message.guild.create_role(name="Muted")
                        for channel in message.guild.channels:
                            await channel.set_permissions(mute_role, speak=False, send_messages=False)

                    await message.author.add_roles(mute_role)
                    await message.author.send(f"@{username}, You have been muted for sending inappropriate messages!")

                    # Unmute after the duration
                    timeout_duration = datetime.timedelta(hours=0.1)
                    await asyncio.sleep(timeout_duration.total_seconds())
                    await message.author.remove_roles(mute_role)

                #warn user
                elif user_violations[user_id] >= threshold_warning:
                    await message.author.send(f"""@{username}, Your message was flagged as innapropriate! If you wish to appeal this decision, 
                                            please type !appeal in the chat. You have {threshold_ban - user_violations[user_id]} violations 
                                            left before you are banned.""")


            await bot.process_commands(message)

        @bot.command()
        async def appeal(ctx):
            await ctx.send('This feature is not yet implemented.')

        bot.run(self.token)




