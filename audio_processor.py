import pyaudio
import numpy as np
import webrtcvad
from collections import deque
from speech_recognition import transcribe_audio

# Parameters for live audio and VAD
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
VAD_SENSITIVITY = 3
SILENCE_THRESHOLD = 2

def process_live_audio(output_text):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    vad = webrtcvad.Vad(VAD_SENSITIVITY)
    audio_buffer = deque(maxlen=int(RATE / CHUNK * SILENCE_THRESHOLD))
    recording = True
    speech_detected = False

    output_text.insert('end', "Listening...\n")

    while recording:
        data = stream.read(CHUNK)
        audio_buffer.append(np.frombuffer(data, dtype=np.int16))

        is_speech = vad.is_speech(data, RATE)

        if is_speech:
            speech_detected = True
        elif speech_detected and not is_speech:
            silence_count = sum(not vad.is_speech(frame.tobytes(), RATE) for frame in audio_buffer)
            if silence_count == len(audio_buffer):
                output_text.insert('end', "Silence detected. Stopping recording...\n")
                recording = False

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Process the recorded audio buffer
    audio_input = np.hstack(audio_buffer)
    transcription = transcribe_audio(audio_input)
    output_text.insert('end', f"Transcription: {transcription}\n")
