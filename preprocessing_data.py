# import used libraries in this code
import os
import python_speech_features as pf
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# ### LOADING THE VOICE DATA FOR VISUALIZATION ###
random_sample = "all_data/background_sound/30.wav"
data, sample_rate = librosa.load(random_sample)
print(data.shape)

# #### VISUALIZING WAVE FORM ##
plt.title("Wave Form")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()

# #### VISUALIZING filter_banks #######
filter_banks,power = pf.fbank(signal=data,samplerate=sample_rate,winlen=0.1,winstep=0.1,
      nfilt=20,nfft=7100,lowfreq=0,highfreq=None,preemph=0.97)
print("Shape of filter bank matrix:", filter_banks.shape)
fbank_mean = np.mean(filter_banks, axis=0) ## calculating mean
fbank_norm = normalize(filter_banks-fbank_mean,axis = 0) #Normalizing Data
derivative1 = np.gradient(fbank_norm,axis=1)
derivative2 = np.gradient(derivative1,axis = 1)
plt.title("filter banks")
librosa.display.specshow(filter_banks, sr=sample_rate, x_axis='time')
plt.show()

# #### Doing this for every sample ##

features_dataset = []           # the dataset to be used in training, testing
fbank_norm_features = []        # extracted filter banks will be stored here, then appended to features dataset
derivative1_features = []       # extracted 1st derivatives will be stored here, then appended to features dataset
derivative2_features = []       # extracted 2nd derivatives will be stored here, then appended to features dataset
labelled_set = []               # set of data labels (1 has WUW) (0 is irrelevant)

# getting all data either locally recorded or from open source datasets
data_path_dict = {
    "down"  : ["all_data/mini_speech_commands/down/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/down/")],
    "left"  : ["all_data/mini_speech_commands/left/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/left/")],
    "no"    : ["all_data/mini_speech_commands/no/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/no/")],
    "right" : ["all_data/mini_speech_commands/right/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/right/")],
    "stop"  : ["all_data/mini_speech_commands/stop/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/stop/")],
    "up"    : ["all_data/mini_speech_commands/up/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/up/")],
    "yes"   : ["all_data/mini_speech_commands/yes/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/yes/")],
    "random": ["all_data/background_sound/" + file_path for file_path in os.listdir("all_data/background_sound/")],
    "go1"   : ["all_data/mini_speech_commands/go/" + file_path for file_path in os.listdir("all_data/mini_speech_commands/go/")],
    "go2"   : ["all_data/go_data/" + file_path for file_path in os.listdir("all_data/go_data/")]
}

# the background_sound/ directory has all sounds which DOES NOT CONTAIN wake word (0)
# the go_data/ directory has all sound WHICH HAS Wake word (1)
# make the dataset dictionary

dataset_dict = {
    0: [],
    1: []
}

# fill the dataset dictionary based on wakeup word present or not

for file in data_path_dict["go1"]:
    dataset_dict[1].append(file)
#for file in data_path_dict["go2"]:
#    dataset_dict[1].append(file)
for file in data_path_dict["down"]:
    dataset_dict[0].append(file)
for file in data_path_dict["left"]:
    dataset_dict[0].append(file)
#for file in data_path_dict["no"]:
#    dataset_dict[0].append(file)
#for file in data_path_dict["right"]:
#    dataset_dict[0].append(file)
#for file in data_path_dict["stop"]:
#    dataset_dict[0].append(file)
#for file in data_path_dict["up"]:
#    dataset_dict[0].append(file)
#for file in data_path_dict["yes"]:
#    dataset_dict[0].append(file)
#for file in data_path_dict["random"]:
#    dataset_dict[0].append(file)


# running through all audio files and extracting features
# the paper stated features extracted include:
# filter banks (normalized)
# 1st and 2nd derivatives of the filter banks

# note that extracting the selected features takes time, so don't worry the program didn't crash
# approximately 5 minutes

for class_label, list_of_files in dataset_dict.items():         # for each list in dataset dictionary (0 and 1)
    for single_file in list_of_files:                           # for each file in the list, extract the following
        audio, sample_rate = librosa.load(single_file)          # Loading file
        audio = librosa.util.fix_length(audio, size=len(data))  # Padding data to same length
        filter_bank, power = pf.fbank(signal=audio, samplerate=sample_rate, winlen=0.1, winstep=0.1,
      nfilt=20, nfft=7100, lowfreq=0, highfreq=None, preemph=0.97)  # # Applying filter bank feature extraction
        # Note that filter bank parameters here are different from those made in the paper.
        # This is done to reduce the dataset so that training is not long. Differences are stated in ReadMe file.

        fbank_mean = np.reshape(np.mean(filter_bank, axis=1), (-1, 1))   # calculating mean
        derivative1 = np.gradient(filter_bank, axis=0)                    # Calculating 1st derivative
        derivative2 = np.gradient(derivative1, axis=0)                   # Calculating 2nd derivative
        fbank_norm = normalize(filter_bank - fbank_mean, axis=1)  # Normalizing Data
        derivative1 = normalize(derivative1, axis=1)
        derivative2 = normalize(derivative2, axis=1)

        for window in range(len(fbank_norm)):                            # Storing data in designated list
            for channel in range(len(fbank_norm[window])):
                fbank_norm_features.append(fbank_norm[window, channel])
                derivative1_features.append(derivative1[window, channel])
                derivative2_features.append(derivative2[window, channel])

            labelled_set = np.append(labelled_set, class_label)

    print(f"Info: Succesfully Preprocessed Class Label {class_label}")

# Reshaping extracted features to be alligned (each window is a row. each channel is a column) shape = (x,20)
fbank_norm_features = np.reshape(fbank_norm_features, (-1, 20))
derivative1_features = np.reshape(derivative1_features, (-1, 20))
derivative2_features = np.reshape(derivative2_features, (-1, 20))
labelled_set = np.reshape(labelled_set, (-1, 1))

# Appending all data into the final list to be exported into csv file
features_dataset = np.append(derivative2_features, labelled_set, axis=1)
features_dataset = np.append(derivative1_features, features_dataset, axis=1)
features_dataset = np.append(fbank_norm_features, features_dataset, axis=1)

# Exporting CSV File:
# Naming Columns of the CSV File to be accessed later when extracting data
column_names = []
for i in range(1, 1+len(fbank_norm_features.T)):
    column_names.append(F"filter bank {i}")
for i in range(1, 1+len(derivative1_features.T)):
    column_names.append(F"1st derivative {i}")
for i in range(1, 1+len(derivative2_features.T)):
    column_names.append(F"2nd derivative {i}")

column_names.append("labels")

df = pd.DataFrame(features_dataset, columns=column_names)
# #### SAVING FOR FUTURE USE ###
df.to_csv("final_audio_data_csv/audio_data1.csv")
