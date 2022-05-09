######## IMPORTS ##########
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import python_speech_features as pf
import numpy as np
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model

####### ALL CONSTANTS #####
fs = 256000
seconds = 1
filename = "prediction.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/WWD.h5")

print("Prediction Started: ")
i = 0
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)
    audio, sample_rate = librosa.load(filename)
    filter_bank, power = pf.fbank(signal=audio, samplerate=sample_rate, winlen=0.1, winstep=0.1,
                                  nfilt=20, nfft=7100, lowfreq=0, highfreq=None,
                                  preemph=0.97)  # # Applying filter bank feature extraction
    # Note that filter bank parameters here are different from those made in the paper.
    # This is done to reduce the dataset so that training is not long. Differences are stated in ReadMe file.

    fbank_mean = np.reshape(np.mean(filter_bank, axis=1), (-1, 1))  # calculating mean
    derivative1 = np.gradient(filter_bank, axis=0)  # Calculating 1st derivative
    derivative2 = np.gradient(derivative1, axis=0)  # Calculating 2nd derivative
    fbank_norm = normalize(filter_bank - fbank_mean, axis=1)  # Normalizing Data
    derivative1 = normalize(derivative1, axis=1)
    derivative2 = normalize(derivative2, axis=1)

    X_input = np.reshape(np.append(fbank_norm, np.append(derivative1, derivative2, axis=1), axis=1), (1, 10, 60))
    prediction = model.predict(X_input)
    if prediction > 0.85:
        print(f"Wake Word Detected for ({i})")
        print("Confidence:", prediction)
        i += 1

    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", 1-prediction)
