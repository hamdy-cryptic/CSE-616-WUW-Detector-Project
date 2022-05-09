# CSE-616-WUW-Detector-Project
  Wake Word Detection (also known as Hot word detection) is a technique mainly used in ChatBots to wake them. 'Okay Google', 'Siri' and 'Alexa' are the wake words used by Google assistant, Apple and Amazon's Alexa respectively. This project aims to use Neural Network Architectures to develop a wake up word detector. The detector is inspired from the work of Shilei Zhang and Pritish Mishra. The aim of this project is to test the architecture used in Wake up word detection of a recent paper and test its viability. Important references are:

1- Pritish Mishra wake up word detection: https://github.com/pritishmishra703/WakeWordDetection

The work of Mishra inspired the workflow and file usage of this project, but different implementations of Neural network and dataset is used.

2- Wake-up-word spotting using end-to-end deep neural network system published by Shilei Zhang, Wen Liu, Yong Qin https://ieeexplore.ieee.org/document/7900073

  This paper inspired the main neural network architecture. Some changes are found between the paper implementation and the one found in the repo because of using different datasets. Further discussion about the differences will be seen later. The main core of the architecture is similar in both implementations as bidirectional LSTMs (5 layers) are used, and extracted features are the same too. 

# File usage
  After downloading a clone of this repo, you can run the **main.py** file to train the model based on the csv file found in the *final_audio_data_csv* folder. When training finishes, you can run the **real_time_testing.py** to test the trained model with your own voice. The trained model is saved in the folder *saved_model* and can be used later to test real time data. This means that you will not need to train the model every time unless you changes hyperparameters and want to re train the model. Please note that you must run train the model at least once because this repo doesn't contain the trained model. This is done to let you tune the model as you wish and modify the training data before finalizing the model.

## data_gathering.py
  This file is used to download the mini-speech-commands dataset created by Google. Audio samples in this dataset are each 1 second long, and they are labeled in a folder name indicating the word being said in the audio file. This dataset is used in this implementation to train the neural network on identifying a special wakeword "go" and consider anything else as garbage. The dataset will be downloaded in a folder named *all_data* and it will be downloaded the first run of the script only, so no overwriting will occur when multiple run instances are initated.
 
  The .py file also enables you to make your own dataset to be used in training. The default values used in this code are compatible with the mini-speech-commands dataset so you can train your model using data from both datasets simultaneously. This feature enables you to increase the word pool from which you train the neural network and not just rely on words specified in the google dataset. The function description is as follows and referenced from Pritish Mishra.
  
  record_audio_and_save():- It records a audio of 2 seconds of you saying the Wake Word. It takes two parameters namely n_times and save_path. n_times is 'How many times it should record you saying the Wake Word?'. Default is set to 50. In save_path you have to provide the path to the directory where it can store generated .wav files.

record_background_sound():- It records a audio of 2 seconds of the background sounds. It takes two parameters namely n_times and save_path. n_times is 'How many times it should record the backgound sounds?'. Default is set to 50. In save_path you have to provide the path to the directory where it can store generated .wav files.

*_Note:_* You should create folders in which the .py file will store the audio files. The default files you should create should be located in the *all_data* folder and be named *go_data* and *background_data* to store wakeup word audio "go" and background noise respectively. You can rename or change the location of the folders you will create by modifying the code.

## preprocessing_data.py
