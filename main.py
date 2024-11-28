from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Callable
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    LiveResultResponse
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
                    {"role": "system", "content": "You are Adriana, A nice empathic friend, bit flirty and cute. Start the conversation with your introduction.This conversation is likely a phone call. \n**Note**: Do not generate the expressions."},
                ]
    is_finals= []
    await websocket.accept()

    try:
        # Initialize the Deepgram connection (example client usage)
        dg_connection = dg_client.listen.websocket.v("1")
        client = ElevenLabs(api_key=os.environ.get("ELEVEN_API_KEY"))
        async def send_data(sentence):
            await websocket.send_text(
                f"<b>You:</b> {sentence}<br>"
            )
            st= False
            nonlocal messages
            prompt= {"role": "user", "content": sentence}
            messages.append(prompt)
            if not st:
                completion = openai_reasoning_agent(messages=messages)
                reasoning_response =    completion.choices[0].message.content.strip()
                response= {"role": "assistant", "content": reasoning_response}
                messages.append(response)
                await websocket.send_text(
                    f"<b> Adri:</b> {reasoning_response} <br>"
                )
                audio= client.generate(
                    text=reasoning_response, 
                    model="eleven_turbo_v2_5",
                    voice="tTQzD8U9VSnJgfwC6HbY",
                    # voice="zT03pEAEi0VHKciJODfn",
                    # voice="amiAXapsDOAiHJqbsAZj",
                    stream=True,
                    # voice_settings=VoiceSettings()
                )
                play(audio)
            else:
                completion = openai_reasoning_agent(messages=messages, stream=st)
                collected_chunks = []
                collected_messages = []
                await websocket.send_text(
                    f"<b> Adri:</b> "
                )
                # iterate through the stream of events
                for chunk in completion:
                    collected_chunks.append(chunk)  # save the event response
                    chunk_message = chunk.choices[0].delta.content  # extract the message
                    await websocket.send_text(chunk_message)
                    audio= client.generate(
                    text=chunk_message, 
                    model="eleven_turbo_v2_5",
                    voice="tTQzD8U9VSnJgfwC6HbY",
                    stream=True,
                    # voice_settings=VoiceSettings()
                )
                    play(audio)
                    collected_messages.append(chunk_message)  # save the message
                await websocket.send_text(" <br>")

                # print the time delay and text received

                collected_messages = [m for m in collected_messages if m is not None]
                full_reply_content = ''.join(collected_messages)
                response= {"role": "assistant", "content": full_reply_content}
                messages.append(response)

            print(messages)
            print("#"*20)

        def on_message(self, result:LiveResultResponse, *args, **kwargs):
            nonlocal is_finals
            sentence = result.channel.alternatives[0].transcript.strip()
            
            if len(sentence)<=0 and len(is_finals)==0:
                return
            if not len(sentence)==0:
                is_finals.append(sentence)
            if not result.speech_final:
                print(result.speech_final, sentence)
                return  # Exit early if the speech is not final

            
            sentence= " ".join(is_finals)
            is_finals= []
            print(f"Final transcript: {sentence}")  # Debugging log
            asyncio.run(send_data(sentence))
            

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # Connect to the websocket with desired options
        options = LiveOptions(
            model="nova-2", 
            language="multi", 
            punctuate=True, 
            filler_words=True, 
            vad_events=True,
            # Time in milliseconds of silence to wait for before finalizing speech
            endpointing=1000,
            )
        print("\n\nPress Enter to stop recording...\n\n")
        if dg_connection.start(options) is False:
            print("Failed to start connection")
            return

        # Handle incoming data
        while True:
            data = await websocket.receive_bytes()
            c=  dg_connection.send(data)
            # print(c)


    except Exception as e:
        print(f'Could not process audio: {e}')
    finally:
        await websocket.close()
        print(messages)

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)