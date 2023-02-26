from revChatGPT.V1 import Chatbot
import json

# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

chatbot = Chatbot(config=config)

def get_chatbot_response(prompt):
    response = ""

    for data in chatbot.ask(prompt):
        response = data["message"]

    return response
