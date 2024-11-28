import webrtcvad
import collections
import numpy as np
import asyncio

async def websocket_endpoint(websocket: WebSocket):
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Adjust aggressiveness: 0 (low) to 3 (high)

    # Settings for silence detection
    sample_rate = 16000  # Assuming 16 kHz audio
    frame_duration_ms = 30  # Frame size in ms
    chunk_size = int(sample_rate * frame_duration_ms / 1000) * 2  # Bytes per frame
    silence_threshold = 1.0  # Seconds of silence to consider speech ended

    ring_buffer = collections.deque(maxlen=int(silence_threshold / (frame_duration_ms / 1000)))
    speech_detected = False
    buffer_audio = b''

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()
            buffer_audio += data

            # Process the audio in chunks
            while len(buffer_audio) >= chunk_size:
                chunk = buffer_audio[:chunk_size]
                buffer_audio = buffer_audio[chunk_size:]

                is_speech = vad.is_speech(chunk, sample_rate)
                ring_buffer.append(is_speech)

                # Check for end of speech
                if any(ring_buffer):  # Speech detected
                    speech_detected = True
                elif speech_detected:  # Silence detected after speech
                    # Finalize speech segment
                    speech_detected = False
                    audio_segment = buffer_audio[:chunk_size]
                    await process_audio(audio_segment, websocket)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


async def process_audio(audio_segment, websocket):
    """
    Handles audio processing after speech detection is complete.
    """
    # Convert audio to text using transcription service (e.g., Deepgram)
    transcript = await transcribe_audio(audio_segment)
    await websocket.send_text(f"<b>You:</b> {transcript}<br>")

    # Generate AI response
    response = openai_reasoning_agent([{"role": "user", "content": transcript}])
    assistant_reply = response.choices[0].message.content.strip()
    await websocket.send_text(f"<b>Adri:</b> {assistant_reply}<br>")

    # Convert response to speech and play
    audio = ElevenLabs(api_key=os.environ.get("ELEVEN_API_KEY")).generate(
        text=assistant_reply, voice="tTQzD8U9VSnJgfwC6HbY", stream=True
    )
    play(audio)


async def transcribe_audio(audio_segment):
    """
    Example function to handle audio transcription using Deepgram.
    """
    # Send `audio_segment` to Deepgram for transcription
    return "transcribed text"  # Placeholder
