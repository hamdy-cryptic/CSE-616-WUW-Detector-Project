# CSE-616-WUW-Detector-Project
This project aims to use Neural Network Architectures to develop a wake up word detector. The detector is inspired from the work of Shilei Zhang and Pritish Mishra. The aim of this project is to test the architecture used in Wake up word detection of a recent paper and test its viability. Important references are:

1- Pritish Mishra wake up word detection: https://github.com/pritishmishra703/WakeWordDetection

The work of Mishra inspired the workflow and file usage of this project, but different implementations of Neural network and dataset is used.

2- Wake-up-word spotting using end-to-end deep neural network system published by Shilei Zhang, Wen Liu, Yong Qin https://ieeexplore.ieee.org/document/7900073

This paper inspired the main neural network architecture. Some changes are found between the paper implementation and the one found in the repo because of using different datasets. Further discussion about the differences will be seen later. The main core of the architecture is similar in both implementations as bidirectional LSTMs (5 layers) are used, and extracted features are the same too. 

# File usage
After downloading a clone of this repo, you can run the main.py file to train the model based on the csv file found in the final_audio_data_csv folder. When training finishes, you can run the real_time_testing.py to test the trained model with your own voice.

$ the first file:
