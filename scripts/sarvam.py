import requests
import os
import pyaudio
import wave
import base64
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")

# Recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"


def initialize_audio():
    """Initialize PyAudio instance"""
    return pyaudio.PyAudio()


def start_recording(p):
    """Start recording audio"""
    return p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


def stop_recording(stream, p):
    """Stop recording and terminate audio stream"""
    stream.stop_stream()
    stream.close()
    p.terminate()


def save_audio(frames, p, filename=WAVE_OUTPUT_FILENAME):
    """Save recorded frames to a WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return filename


def record_audio():
    """Main function to record audio and save it to a file"""
    p = initialize_audio()
    stream = start_recording(p)

    print("Recording...")
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
    print("Finished recording.")

    stop_recording(stream, p)
    return save_audio(frames, p)


def transcribe_audio(audio_file_path):
    """Transcribe audio using SARVAM AI's Speech-to-Text API"""
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
    """Record and transcribe audio"""
    audio_file_path = record_audio()
    transcription = transcribe_audio(audio_file_path)

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    return transcription


def play_audio(decoded_audio):
    """Play audio from a decoded byte stream"""
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
    """Fetch audio from SARVAM AI's Text-to-Speech API"""
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
    """Convert text to speech and play the audio"""
    audio_clips = fetch_text_to_speech_audio(text)
    for audio_clip in audio_clips:
        decoded_audio = base64.b64decode(audio_clip)
        play_audio(decoded_audio)


# Example usage
if __name__ == "__main__":
    # Transcribe recorded audio
    transcription = record_and_transcribe()
    print(f"Transcribed text: {transcription}")

    # # Convert text to speech
    # text_to_speech("This is amazing, ye bhot jyada hi amazing hai")
