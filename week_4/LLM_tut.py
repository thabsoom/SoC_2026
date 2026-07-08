import chainlit as cl
import openai 
import os 
os.environ['OPENAI_API_KEY']="secret"

# return everything that the user inputs. 
# pass the message into chatgpt api .send() the answer
from openai import OpenAI
client = OpenAI()
@cl.on_message
async def main(message: cl.Message):
    response=client.completions.create(
        model="gpt-4",
        messages= [
            {"role":"assistant","content":"you are a helpful assistant"},
            {"role":"user","content":message.content}
        ],
        temperature= 1
    )
    await cl.Message(content=f"{response['choices'][0]['message']['content']}",).send()