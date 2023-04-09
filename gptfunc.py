from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os
import json


# read config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# get openai api key from config.json
api_key = config["openai_api_key"]

os.environ["OPENAI_API_KEY"] = api_key



def get_chatbot_response(text,model_name="gpt-3.5-turbo"):
    template="你是一位十分善于解决问题、按步骤思考的专业咨询顾问。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "针对以下监管要求编写详细的审计步骤、访谈问题和资料清单：{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#     messages = [
#     SystemMessage(content= "你是一位十分善于解决问题、按步骤思考的专业咨询顾问。"),
#     # combine the prompt and the instruction with the end-of-sequence token
#     HumanMessage(content=  "针对以下监管要求编写详细的审计步骤："+prompt),
# ]
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chat = ChatOpenAI(model_name=model_name)#, max_tokens=512)

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    # response = chat(chat_prompt.format_prompt(text=text).to_messages())
    response = chain.run(text=text)
    return response
