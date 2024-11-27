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
from elevenlabs import play, stream, VoiceSettings
from elevenlabs.client import ElevenLabs
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


@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket,):
    messages = [
                    {"role": "system", "content": "You are Adriana, A nice empathic friend, bit flirty and cute. Start the conversation with your introduction.You usually talk in  hinglish ,  based on the flow of conversation.This conversation is likely a phone call. \n**Note**: Do not generate the expressions."},
                ]
    await websocket.accept()

    try:
        # Initialize the Deepgram connection (example client usage)
        dg_connection = dg_client.listen.websocket.v("1")
        client = ElevenLabs(api_key=os.environ.get("ELEVEN_API_KEY"))
        async def send_data(sentence):
            await websocket.send_text(
                f"<b>You:</b> {sentence}<br>"
            )
            nonlocal messages
            prompt= {"role": "user", "content": sentence}
            messages.append(prompt)
            response, messages = openai_reasoning_agent(messages=messages)

            await websocket.send_text(
                f"<b> Adri:</b> {response} <br>"
            )
            audio= client.generate(
                text=response, 
                model="eleven_turbo_v2_5",
                # voice="XB0fDUnXU5powFXDhCwa",
                voice="zT03pEAEi0VHKciJODfn",
                # voice="amiAXapsDOAiHJqbsAZj",
                stream=True,
                # voice_settings=VoiceSettings()
            )
            play(audio)
            print(messages)
            print("#"*20)

        def on_message(self, result, *args, **kwargs):
            # print(args, kwargs)
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            # print(f"speaker: {sentence}")
            asyncio.run(send_data(sentence))
            return sentence

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # Connect to the websocket with desired options
        options = LiveOptions(model="nova-2", language="hi", punctuate=True, filler_words=True)
        print("\n\nPress Enter to stop recording...\n\n")
        if dg_connection.start(options) is False:
            print("Failed to start connection")
            return

        dg_connection.send("start")

        # Handle incoming data
        while True:
            data = await websocket.receive_bytes()
            dg_connection.send(data)


    except Exception as e:
        raise Exception(f'Could not process audio: {e}')
    finally:
        await websocket.close()
        print(messages)

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)