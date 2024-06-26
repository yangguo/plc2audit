import json
import os

import pandas as pd
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

import openai

# openai.api_base="https://tiny-shadow-5144.vixt.workers.dev/v1"
# openai.api_base="https://super-heart-4116.vixt.workers.dev/v1"
openai.api_base = "https://az.139105.xyz/v1"

# llm = ChatOpenAI(model_name=model_name )
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=AZURE_API_KEY)


# use azure model
# llm = AzureChatOpenAI(
#     openai_api_base=AZURE_BASE_URL,
#     openai_api_version="2023-03-15-preview",
#     deployment_name=AZURE_DEPLOYMENT_NAME,
#     openai_api_key=AZURE_API_KEY,
#     openai_api_type = "azure",
# )
# use cohere model
# llm = Cohere(model="command-xlarge-nightly",cohere_api_key=COHERE_API_KEY,temperature=0)


def get_chatbot_response(text, model_name="gpt-3.5-turbo"):
    template = "你是一位十分善于解决问题、按步骤思考的专业咨询顾问。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = f"""
    针对监管要求编写审计步骤及相关内容：
    - '步骤编号'：'每一项审计工作的编号'
    - '审计工作'：'针对每一项监管要求，需要进行的具体审计工作'
    - '访谈问题'：'针对每一项审计工作，需要向被审计方提出的访谈问题'
    - '资料清单'：'针对每一项审计工作，需要准备的审计资料'

    监管要求使用'''进行间隔。

    输出的格式应为JSON列表，每个列表项为一个对象，包含以上四个字段："步骤编号"，"审计工作"，"访谈问题"，"资料清单"。字段的值应为具体的信息。

    以下是一个包含一个对象的输出样例：
    [{{{{"步骤编号": "1",
    "审计工作": "确认证券期货机构是否有相关文档分类管理制度",
    "访谈问题": "请描述证券期货机构的文档分类管理制度",
    "资料清单": "证券期货机构的文档分类管理制度" }}}}]

    监管要求：'''{text}'''
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # chat = ChatOpenAI(model_name=model_name)#, max_tokens=512)

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    # response = chat(chat_prompt.format_prompt(text=text).to_messages())
    response = chain.run(text=text)
    return response
    # convert json to dataframe
    # df = convert_json_to_df(response)
    # return df


# convert json response to dataframe
def convert_json_to_df(response):
    data_dict = json.loads(response)

    # Convert dictionary to DataFrame
    # df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = pd.DataFrame.from_dict(data_dict)

    return df


def merge_items(data):
    data_dict = json.loads(data)
    merged = {"审计工作": [], "访谈问题": [], "资料清单": []}

    for item in data_dict:
        for key in merged:
            merged[key].append(item["步骤编号"] + "." + item[key])

    for key in merged:
        merged[key] = ";".join(merged[key])

    return merged


def separate_items(input_list):
    audit_work_list = []
    interview_question_list = []
    document_list = []

    for item in input_list:
        audit_work_list.append(item["审计工作"])
        interview_question_list.append(item["访谈问题"])
        document_list.append(item["资料清单"])

    return audit_work_list, interview_question_list, document_list


def extract_information_from_case(text, model_name="gpt-3.5-turbo"):

    response_schemas = [
        ResponseSchema(name="当事人", description="处罚案件的当事人名称，例如：ABC公司"),
        ResponseSchema(name="违法违规事实", description="处罚案件的具体违法违规事实，例如：ABC公司违反了证券交易规定，從事内幕交易"),
           ResponseSchema(name="处罚结果", description="处罚案件的详细处罚结果，例如：警告、罚款10000元"),
        ResponseSchema(name="处罚依据", description="处罚案件的处罚依据条款，例如：《证券法》第一百八十九条第一款"),
       ResponseSchema(name="监管部门", description="处罚案件的监管部门，例如：中国证监会"),
        ResponseSchema(name="处罚时间", description="处罚案件的处罚时间，例如：2023年2月27日"),
       ResponseSchema(name="违法违规类型", description="处罚案件的违法违规类型，例如：内幕交易"),
        ResponseSchema(name="罚款总金额", description="处罚案件的罚款总金额，例如：10万元"),
    ResponseSchema(name="没收总金额", description="处罚案件的没收总金额，例如：10万元")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    template = """
    你是一位具有高级文本理解能力的人工智能助手，你的任务是从监管处罚案件的描述中提取关键信息。
    
    我需要你从下面的处罚案件文本中提取以下信息：当事人、违法违规事实、处罚结果、处罚依据条款、监管部门、处罚时间、违法违规类型、罚款总金额、没收总金额。
    
    {format_instructions}

    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = f"""    
    处罚案件的内容如下:
    {text}
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate(
        messages=[system_message_prompt, human_message_prompt],
         input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
    )

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(text=text)

    json_response=output_parser.parse(response)

    return json_response
