import subprocess

# Execute audio_capture.py
capture_process = subprocess.Popen(['python', 'audio_capture.py'], shell=True)
try:
    capture_process.wait()  # Wait for audio_capture.py to finish executing
except KeyboardInterrupt:
    pass  # Ignore KeyboardInterrupt and continue execution

# Execute whisper <name.wav>
whisper_process = subprocess.Popen(['whisper', 'output.wav'], shell=True)
whisper_process.wait()  # Wait for whisper command to finish executing

# Execute logistic_regression.py
regression_process = subprocess.Popen(['python', 'logistic_regression.py', 'path/to/transcribed.txt'], shell=True)
regression_process.wait()  # Wait for logistic_regression.py to finish executing
