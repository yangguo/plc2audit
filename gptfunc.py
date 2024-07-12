import json
import os
import re

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain import LLMChain, hub
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.output_parsers import (
    ResponseSchema,
    RetryWithErrorOutputParser,
    StructuredOutputParser,
)
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, StrOutputParser, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

load_dotenv()

AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")
AZURE_DEPLOYMENT_NAME_GPT4_TURBO = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_TURBO")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# import openai

# openai.api_base="https://super-heart-4116.vixt.workers.dev/v1"
# openai.api_base="https://tiny-shadow-5144.vixt.workers.dev/v1"
# openai.api_base = "https://az.139105.xyz/v1"

# llm = ChatOpenAI(model_name="gpt-3.5-turbo")
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=AZURE_API_KEY)

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

# convert gpt model name to azure deployment name
gpt_to_deployment = {
    "gpt-35-turbo": AZURE_DEPLOYMENT_NAME,
    "gpt-35-turbo-16k": AZURE_DEPLOYMENT_NAME_16K,
    "gpt-4": AZURE_DEPLOYMENT_NAME_GPT4,
    "gpt-4-32k": AZURE_DEPLOYMENT_NAME_GPT4_32K,
    "gpt-4-turbo": AZURE_DEPLOYMENT_NAME_GPT4_TURBO,
}


# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_BASE_URL,
        openai_api_version="2023-12-01-preview",
        azure_deployment=deployment_name,
        openai_api_key=AZURE_API_KEY,
    )
    return llm


def get_chatbot_response(text, model_name="gpt-35-turbo"):
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
    llm = get_azurellm(model_name)
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


def extract_information_from_case(text, model_name="gpt-35-turbo"):
    response_schemas = [
        ResponseSchema(name="当事人", description="处罚案件的当事人名称，例如：ABC公司"),
        ResponseSchema(
            name="违法违规事实", description="处罚案件的具体违法违规事实，例如：ABC公司违反了证券交易规定，從事内幕交易"
        ),
        ResponseSchema(name="处罚结果", description="处罚案件的详细处罚结果，例如：警告、罚款10000元"),
        ResponseSchema(name="处罚依据", description="处罚案件的处罚依据条款，例如：《证券法》第一百八十九条第一款"),
        ResponseSchema(name="监管部门", description="处罚案件的监管部门，例如：中国证监会"),
        ResponseSchema(name="处罚时间", description="处罚案件的处罚时间，例如：2023年2月27日"),
        ResponseSchema(name="违法违规类型", description="处罚案件的违法违规类型，例如：内幕交易"),
        ResponseSchema(name="罚款总金额", description="处罚案件的罚款总金额，例如：10万元"),
        ResponseSchema(name="没收总金额", description="处罚案件的没收总金额，例如：10万元"),
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
        partial_variables={"format_instructions": format_instructions},
    )

    # print(chat_prompt)
    llm = get_azurellm(model_name)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(text=text)
    return response

    # json_response=output_parser.parse(response)

    # return json_response


def shorten_and_summarize(text, model_name="gpt-35-turbo"):
    token_limit = 7700
    part_limit = 3000

    # Convert text to tokens using the rough approximation
    tokens = count_tokens(text)
    print("tokens: ", tokens)

    # If the text is already within the token limit, return as is
    if tokens <= token_limit:
        return text

    # If the text exceeds the token limit, split and summarize
    first_part = text[:part_limit]
    last_part = text[-part_limit:]

    print("first_part: ", first_part)
    print("last_part: ", last_part)

    # Approximate tokens in the middle part
    middle_part = text[part_limit:-part_limit]
    middle_part_tokens = tokens - 2 * part_limit

    print("middle_part: ", middle_part)
    print("middle_part_tokens: ", middle_part_tokens)

    # Split text into sentences
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    chunks = splitter.split_text(middle_part)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Define prompt template
    prompt_template = """Write a concise summary of the following:

    {text}

    CONCISE SUMMARY IN CHINESE:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    llm = get_azurellm(model_name)
    # Summarize the middle part
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=PROMPT,
        combine_prompt=PROMPT,
    )
    response = chain({"input_documents": docs}, return_only_outputs=True)
    middle_part_summary = response["output_text"]

    # Join all parts together
    result = first_part + "\n" + middle_part_summary + "\n" + last_part
    return result


def text_cleanup(text):
    # convert to utf-8
    text = text.encode("utf-8").decode("utf-8")
    # Remove non-Chinese characters and non-English characters and non-numeric characters
    # reg=r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    # reg=r'[^\u4e00-\u9fffa-zA-Z0-9]'
    # reg=r'[^\u4e00-\u9fa5a-zA-Z0-9.,!?;:\'\"@#\$%\^&\*\(\)\-\+=<>\[\]{}|\\/~_]'
    reg = r"[^\u4e00-\u9fa5\u3000-\u303fa-zA-Z0-9.,!?;:\'\"@#\$%\^&\*\(\)\-\+=<>\[\]{}|\\/~_]"
    text = re.sub(reg, "", text)

    return text


def count_tokens(text):
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    return token_count


def get_audit_steps(text, model_name="gpt-35-turbo"):
    # response_schemas = [
    #     ResponseSchema(name="审计步骤", description="针对监管要求，需要执行的多项具体审计工作步骤"),
    #     ResponseSchema(name="访谈问题", description="针对监管要求，需要向被审计方提出的多项访谈问题"),
    #     ResponseSchema(name="资料清单", description="针对监管要求，需要被审计方准备的多项审计资料"),
    # ]

    # Define your desired data structure.
    # class Auditsteps(BaseModel):
    #     审计工作步骤: str = Field(description="针对监管要求，需要针对向被审计方执行的具体审计工作步骤")
    #     访谈问题: str = Field(description="针对监管要求，需要向被审计方提出的访谈问题")
    #     资料清单: str = Field(description="针对监管要求，需要被审计方准备的审计资料")

    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Set up a parser + inject instructions into the prompt template.
    # parser = JsonOutputParser(pydantic_object=Auditsteps)
    # parser = JsonOutputParser()
    # format_instructions = output_parser.get_format_instructions()

    # template = """
    # 你是一位具有10年资深经验的内部审计师，你的任务是根据监管要求生成审计工作计划。

    # 我需要你根据以下监管要求分解成审计目标，并针对这个审计目标编写详细的审计工作计划，并提供相关内容。内容包括：审计工作步骤、访谈问题、资料清单。

    # 所有的审计步骤、访谈问题和资料清单应当在一个完整的回复中给出。

    # {format_instructions}

    # """

    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # human_template = """
    # 监管要求的内容如下:
    # {text}
    # """

    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # chat_prompt = ChatPromptTemplate(
    #     messages=[system_message_prompt, human_message_prompt],
    #     input_variables=["text"],
    #     partial_variables={"format_instructions": parser.get_format_instructions()},
    # )

    chat_prompt = hub.pull("vyang/get_audit_steps")
    output_parser = StrOutputParser()

    llm = get_azurellm(model_name)

    # chain = LLMChain(llm=llm, prompt=chat_prompt)
    chain = chat_prompt | llm | output_parser
    # response = chain.run(text=text)
    response = chain.invoke({"text": text})

    return response
