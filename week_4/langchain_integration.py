import os

import chainlit as cl
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Load environment variables from .env
load_dotenv()

template = """Question: {question}

Answer: Let's think step by step.
"""


@cl.on_chat_start
async def start():
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )

    llm = OpenAI(
        temperature=1,
        streaming=True,
    )

    # Modern replacement for LLMChain
    chain = prompt | llm

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    response = await chain.ainvoke(
        {"question": message.content},
        config={
            "callbacks": [cl.AsyncLangchainCallbackHandler()]
        }
    )

    await cl.Message(content=response).send()