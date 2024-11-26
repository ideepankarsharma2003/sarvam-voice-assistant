# Example filename: main.py

import httpx
import logging
from deepgram.utils import verboselogs
import threading
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
import os

load_dotenv()

# URL for the realtime streaming audio you would like to transcribe
URL = "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service"


def main():
    try:
        # use default config
        deepgram: DeepgramClient = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"], config = DeepgramClientOptions(
    verbose=logging.WARN,  # Change to logging.INFO or logging.DEBUG for more verbose output
    options={"keepalive": "true"}
))

        # Create a websocket connection to Deepgram
        dg_connection = deepgram.listen.websocket.v("1")

        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            print(f"speaker: {sentence}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # connect to websocket
        options = LiveOptions(model="nova-2")

        print("\n\nPress Enter to stop recording...\n\n")
        if dg_connection.start(options) is False:
            print("Failed to start connection")
            return

        lock_exit = threading.Lock()
        exit = False

        # define a worker thread
        def myThread():
            with httpx.stream("GET", URL) as r:
                for data in r.iter_bytes():
                    lock_exit.acquire()
                    if exit:
                        break
                    lock_exit.release()

                    dg_connection.send(data)

        # start the worker thread
        myHttp = threading.Thread(target=myThread)
        myHttp.start()

        # signal finished
        input("")
        lock_exit.acquire()
        exit = True
        lock_exit.release()

        # Wait for the HTTP thread to close and join
        myHttp.join()

        # Indicate that we've finished
        dg_connection.finish()

        print("Finished")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

if __name__ == "__main__":
    main()


# main.py (python example)
