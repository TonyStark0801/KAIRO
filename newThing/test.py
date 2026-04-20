import mlx_whisper
import sounddevice as sd
import numpy as np
import tempfile
import wave
import os
from scipy.signal import butter, lfilter





# --- CONFIG ---
MODEL_PATH = "./whisper-model" 
FS = 16000  
# If it 'listens forever', INCREASE this (try 0.02 or 0.03)
THRESHOLD = 0.04 
# How many seconds of silence before it processes?
SILENCE_CHUNKS = 10 

def highpass_filter(data, cutoff=300, fs=16000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def record_and_transcribe():
    print(f"\n--- Jarvis Level Meter (M1 Pro) ---")
    print("Watching for voice... (Press Ctrl+C to stop)")
    
    audio_buffer = []
    silent_count = 0
    recording = False

    # 100ms chunks for real-time feel
   # 100ms chunks for real-time feel
    with sd.InputStream(samplerate=FS, channels=1, dtype='float32') as stream:
        while True:
            # 1. READ THE DATA FIRST
            raw_chunk, _ = stream.read(int(0.1 * FS))
            
            # 2. FILTER THE RUMBLE (Flatten to 1D for the filter)
            filtered_chunk = highpass_filter(raw_chunk.flatten()) 
            
            # 3. MEASURE VOLUME OF FILTERED SOUND
            volume = np.max(np.abs(filtered_chunk))
            
            # VISUAL FEEDBACK
            meter_val = int(volume * 100)
            meter = "█" * meter_val
            status = 'REC' if recording else 'WAIT'
            print(f"\rVol: [{meter.ljust(20)}] {status} (Val: {volume:.4f})", end="")

            if volume > THRESHOLD:
                if not recording:
                    recording = True
                    audio_buffer = [] 
                audio_buffer.append(filtered_chunk) # Store the clean audio
                silent_count = 0
            elif recording:
                audio_buffer.append(filtered_chunk)
                silent_count += 1
                
                if silent_count > SILENCE_CHUNKS:
                    print("\n[Processing Sentence...]")
                    process_audio(np.concatenate(audio_buffer))
                    recording = False
                    audio_buffer = []
                    silent_count = 0
                    print("Listening again...")
                    
def process_audio(audio_data):
    # Normalize: Make you loud and clear
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(FS)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        tmp_path = tmp.name

    result = mlx_whisper.transcribe(
        tmp_path, 
        path_or_hf_repo=MODEL_PATH,
        initial_prompt="A software engineer in Mumbai speaking Indian English about coding."
    )
    
    text = result['text'].strip()
    if len(text) > 2:
        print(f"\n>>> {text}")
    
    os.remove(tmp_path)

if __name__ == "__main__":
    try:
        record_and_transcribe()
    except KeyboardInterrupt:
        print("\nOffline.")
