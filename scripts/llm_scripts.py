import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
OPENROUTER_API_KEY= os.environ.get("OPENROUTER_API_KEY")
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)





def openai_reasoning_agent(messages:list, stream=False):
    completion = client.chat.completions.create(
                model="google/gemma-2-9b-it:free",
                messages=messages,
                stream=stream
                )
    return completion

    


