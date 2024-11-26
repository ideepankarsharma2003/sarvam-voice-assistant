from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Callable
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
import asyncio
import threading
import uvicorn
from dotenv import load_dotenv
import os
from scripts import openai_reasoning_agent

load_dotenv()

app = FastAPI()

dg_client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'))

templates = Jinja2Templates(directory="templates")

messages=[
                    {"role": "system", "content": "You are Adriana, A nice empathic friend, bit flirty and cute. Start the conversation with your introduction."},
                    
                ]

@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket,):
    
    await websocket.accept()

    try:
        # Initialize the Deepgram connection (example client usage)
        dg_connection = dg_client.listen.websocket.v("1")

        async def send_data(sentence):
            # global messages
            await websocket.send_text(sentence)
            # sentence, messages= openai_reasoning_agent(messages=messages)

        def on_message(self, result, *args, **kwargs):
            print(args, kwargs)
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            print(f"speaker: {sentence}")
            asyncio.run(send_data(sentence))
            return sentence

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        
        # Connect to the websocket with desired options
        options = LiveOptions(model="nova-2", language="multi")
        print("\n\nPress Enter to stop recording...\n\n")
        if dg_connection.start(options) is False:
            print("Failed to start connection")
            return

        # Handle incoming data
        while True:
            data = await websocket.receive_bytes()
            dg_connection.send(data)
            

    except Exception as e:
        raise Exception(f'Could not process audio: {e}')
    finally:
        await websocket.close()

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)