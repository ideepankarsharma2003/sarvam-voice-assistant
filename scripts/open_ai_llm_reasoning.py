import openai
from openai import OpenAI
import requests
import os
import pyaudio
import wave
import base64
import io
from dotenv import load_dotenv
from tqdm import tqdm
from content import system_prompt

# Load environment variables
load_dotenv()
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set the OpenAI API key
# openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


print(f"{'#'*10} successfully imported all dependencies {'#'*10}")


# Recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"


# Helper Functions (same as before)
def initialize_audio():
    return pyaudio.PyAudio()


def start_recording(p):
    return p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


def stop_recording(stream, p):
    stream.stop_stream()
    stream.close()
    p.terminate()


def save_audio(frames, p, filename=WAVE_OUTPUT_FILENAME):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return filename


def record_audio():
    p = initialize_audio()
    stream = start_recording(p)
    print("Recording...")
    frames = [stream.read(CHUNK) for _ in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS)))]
    # print("Finished recording.")
    stop_recording(stream, p)
    return save_audio(frames, p)


def transcribe_audio(audio_file_path):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    payload = {"language_code": "hi-IN", "model": "saarika:v1"}

    with open(audio_file_path, "rb") as audio_file:
        files = {"file": (audio_file_path, audio_file, "audio/wav")}
        response = requests.post(url, files=files, data=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get("transcript", "")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def record_and_transcribe():
    audio_file_path = record_audio()
    transcription = transcribe_audio(audio_file_path)
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
    return transcription


def play_audio(decoded_audio):
    with wave.open(io.BytesIO(decoded_audio), 'rb') as wf:
        p = initialize_audio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        stop_recording(stream, p)


def fetch_text_to_speech_audio(text):
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("audios", [])
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def text_to_speech(text):
    audio_clips = fetch_text_to_speech_audio(text)
    for audio_clip in audio_clips:
        decoded_audio = base64.b64decode(audio_clip)
        play_audio(decoded_audio)


# Reasoning Agent Function
# def openai_reasoning_agent(user_prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=user_prompt,
#         max_tokens=150,
#         temperature=0.7
#     )
#     reasoning_response = response.choices[0].text.strip()
#     return reasoning_response

def openai_reasoning_agent(messages:list):
    completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
                )

    reasoning_response =    completion.choices[0].message.content.strip()
    resp= {"role": "assistant", "content": reasoning_response}
    messages.append(resp)
    return reasoning_response, messages


def run_voice_assistant():
    messages=[
                    {"role": "system", "content": system_prompt},
                    
                ]
    print("Generating AI Response....")
    ai_response, messages = openai_reasoning_agent(messages)
    print(f"AI Response: {ai_response}")
    text_to_speech(ai_response)

    while True:
        print("Waiting for user input...")
        
        user_input = record_and_transcribe()
        if not user_input:
            print("Could not transcribe audio. Please try again.")
            continue
        
        print(f"User said: {user_input}")

        prompt= {"role": "user", "content": user_input}
        messages.append(prompt)
        print("Generating AI Response....")
        ai_response, messages = openai_reasoning_agent(messages)
        print(f"AI Response: {ai_response}")

        text_to_speech(ai_response)

        if user_input.lower() in ["exit", "quit", "stop"]:
            exit_message= "Exiting the voice assistant. Goodbye!"
            text_to_speech(exit_message)
            print(exit_message)
            break


# Start the reasoning agent
if __name__ == "__main__":
    run_voice_assistant()
