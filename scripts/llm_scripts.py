import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)





def openai_reasoning_agent(messages:list):
    completion = client.chat.completions.create(
                model="meta-llama/llama-3.2-3b-instruct:free",
                messages=messages
                )

    reasoning_response =    completion.choices[0].message.content.strip()
    resp= {"role": "assistant", "content": reasoning_response}
    messages.append(resp)
    return reasoning_response, messages

