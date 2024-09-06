import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

# Load models once, so they can be reused
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
nlu_model = pipeline("sentiment-analysis")

def transcribe_audio(audio_input):
    input_values = processor(audio_input, return_tensors="pt", padding="longest").input_values
    logits = speech_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def process_text_input(user_input, output_text):
    result = nlu_model(user_input)
    output_text.insert('end', f"Text Analysis: {result}\n")
