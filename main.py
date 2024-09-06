import pyaudio # Soundcard audio I/O access library
import wave # Python 3 module for reading / writing simple .wav files
import whisper
import subprocess

# Setup channel info
FORMAT = pyaudio.paInt16 # data type formate
CHANNELS = 2 # Adjust to your number of channels
RATE = 44100 # Sample Rate
CHUNK = 1024 # Block Size
RECORD_SECONDS = 10 # Record time
WAVE_OUTPUT_FILENAME = "file.wav"

# Startup pyaudio instance
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("recording...")
frames = []

# Record for RECORD_SECONDS
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")


# Stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

# Write your new .wav file with built in Python 3 Wave module
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()


def transcribe_audio(audio_file):
    # Load the Whisper model
    model = whisper.load_model("medium")

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # Define output file name and save the transcription
    output_file = "output2.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f'Output: {result.text}')

    # Return the recognized text
    return result.text

# Example usage
recognized_text = transcribe_audio("file.wav")
print("Recognized text:", recognized_text)





