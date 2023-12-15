# Content Moderator Discord Bot

This project uses a DistilBert model I fine tuned to perform binary classification on whether text is innapropriate or not. The model is then used in a discord bot to moderate messages sent in a discord server.

## Instructions for Use

To use the model, first download the model files from [This Google Drive Folder](https://drive.google.com/drive/folders/1MUpmOU9G1g0DljfV35ddwehzG0w1S69W?usp=sharing)

Place the pytorch_model.bin and config.json files in the model folder.

Then, create a file called token.txt in the root directory of the project and paste your discord bot token in it.

Then run the following command to start the bot:
```bash
python3 bot.py
```

### Dependencies:
- transformers
- discord.py
- torch
- pandas



