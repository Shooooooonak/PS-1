import pyaudio
import wave
import os

# Set the audio parameters
chunk = 1024  # Number of frames in each buffer
sample_format = pyaudio.paInt16  # 16-bit resolution
channels = 2  # Stereo
sample_rate = 44100  # Sample rate (in Hz)

# Specify the file name for saving the recorded audio
file_name = "C:\\Users\\admin\\Desktop\\sentiment\\output.wav"

# Create an instance of the PyAudio class
audio = pyaudio.PyAudio()

# Open the audio stream
stream = audio.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    frames_per_buffer=chunk,
                    input=True)

print("Recording started... Press Ctrl+C to stop.")

frames = []  # Initialize an empty list to store audio frames

try:
    while True:
        # Read audio data from the stream
        data = stream.read(chunk)
        frames.append(data)
except KeyboardInterrupt:
    # Recording stopped by the user

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio instance
    audio.terminate()

    print("Recording stopped.")

    # Save the recorded audio to the same .wav file (overwriting the previous content)
    wave_file = wave.open(file_name, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(sample_format))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
