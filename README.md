# PS-1
A sentiment analysing project employing speech to text (STT) models, Machine Learning (ML) algorithms and other libraries
Here is a detailed step-by-step instructions for running the following project on your device:

## 1) Downloading the required packages and dependencies:
The python version for this implementation is 3.10.12. Running the python scripts in the
linked folder requires an active python environment for execution. We used mamba, very
similar to miniconda this purpose. Searching up “conda forge” on GitHub ought to do the
trick (with a convenient youtube video for the necessary steps for installation).
Next up, we input the command “mamba activate base” on the terminal of the Raspberry
Pi to activate the environment in context. This isolates the installed libraries and
packages from the main files of your device to prevent any corruption, misplace of files
and smooth execution. You should notice (base) preceding your current directory in the
terminal. Download the folder from the link and save it in an accessible directory. Use the
command “cd (file path to Sentiment)” on the Raspi terminal to change the directory to
this specific folder (named Sentiment)
Now we will begin the installation of the concerned python libraries. We use pip, a
python package manager in the command “pip install (library name here)” to install the
following onto your python environment:
a. pyaudio
b. opencv-python
c. keyboard
d. -U openai-whisper
e. nltk
f. ffmpeg (use “sudo apt install ffmpeg” for this one)
g. pandas
Lastly, we will require the installation of pytorch to finish up the installation process.
Open up the browser on your raspi and go to the home website of pytorch. Scroll down
to the bottom, you will see several fields that you choose between to install the suitable
version of pytorch. Select Linux, Python and CPU and copy the generated link.
Go back to your terminal (ensure that the python environment is still running), and input
the following command “ wget (paste the copied link)” and hit enter.

## 2) Now we will introduce you to the python files (or scripts) at play for this implementation.
If you open the linked file (Sentiment), you will notice a LOT of files, but let’s bring your
attention to three specific ones, namely audio_capture.py, output.wav and
logistic_regression.py.
a. audio_capture.py: This python script is responsible for taking in the audio input
through a microphone and saving the recorded audio as a .wav file. A text will appear on
the screen that says “ recording has started, press ctrl-C to stop”. You are expected to speak your mind when that pops up and press the mentioned keys to end the stream
and save your audio input. The code for it is as below:

![image](https://github.com/Shooooooonak/PS-1/assets/137979966/895db62c-8ab1-42a1-9ecf-f35fd10a522c)

![image](https://github.com/Shooooooonak/PS-1/assets/137979966/f1f405e6-238f-47f1-a5e3-99f82ba53887)


NOTE: Line 12 incudes the directory that the file was saved on my device, do adjust it
according to yours before implementing

b. output.wav: This is the audio file that contains your inputted speech audio input. If we
were execute our STT model (OpenAI Whisper) on this separately, we would’ve used the
command “ whisper (.wav file name)”. You can save .wav files of your own and play
around to have them transcribed. Whisper is capable of taking upto 30 seconds of audio
and recognising a variety of languages from english to urdu. The specification of the
model used for our implementation is “tiny” to save runtime and suit our constrained
needs. The audio processing does take a bit, be patient. The main folder named
Sentiment will also include a .txt file of the same name as the .wav file; it is the
transcript and comes in handy in the next section.

c. logistic_regression.py: This is where all the magic happens. The .txt named
“output.txt” is fed into our ML model (a logistic regression model, you guessed it!) which
reduces each word in the text to its root word, compares it with its database using
Natural Language Processing (NLP) and predicts a sentiment. This sentiment is then fed
into another chunk of code in the same python script which displays the corresponding
emotion on the screen in the form of a cute little robot face. The code for both of this
mentioned in great detail in section 2. Emotion Detection and 3. The Emotion Display
4) You must be wondering, okay we have 3 different python files executing each phase of
this implementation but how do I automate this? Well we have another python script in
the folder Sentiment which honestly, is all you should need to know about if you just want
to use this project. The script is named run.py. It uses a built in python library named
"subprocess" that automatically executes the aforementioned python scripts provided all
exist in the same directory, which they do. Its pretty concise and easy to understand:


![image](https://github.com/Shooooooonak/PS-1/assets/137979966/54daa26a-cd4d-4deb-a34d-bc052abea361)



The moral of this manual is that if you want to run the necessary programs, simply set
your directory to the linked file (Sentiment) and slap the command “python runner.py” in the
terminal and watch how it does its thing.

If the predicted sentiment returns positive, the program will return the image of a smiling face.
The program uses the OpenCV Python library. If the predicted sentiment is negative, then we get an image of a robot frowning. We can also add the
time after which we want the image to disappear using the wait key. But here we have added a
function such that the image disappears when we press ‘q’. 

![image](https://github.com/Shooooooonak/PS-1/assets/137979966/b251f10c-a169-4728-8c8f-c0feca31fe91)

![image](https://github.com/Shooooooonak/PS-1/assets/137979966/f4900b2d-c594-4317-a5da-f6a43338e5f3)



