# import used libraries in this code

import pathlib
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
from tensorflow import keras


###########################################################################
# getting the open source dataset "mini speech commands" made by google
# https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Speech_Commands.ipynb
# codes in this snippet are referenced from the previous link

DATASET_PATH = 'all_data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='all_data')
###########################################################################


##########################################################################
# https://github.com/pritishmishra703/WakeWordDetection
# codes in this snippet are inspired from the previous link and modified to suit the mini speech commands dataset

# Preparing data to be used in the training
# this data is recorded locally using laptop's microphone

def record_audio_and_save(save_path, n_times=50):
    """
    This function will run `n_times` and everytime you press Enter you have to speak the wake word
    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.
    save_path: str
        Where to save the wav file which is generated in every iteration.
    """

    input("To start recording Wake Word press Enter: ")
    for i in range(n_times):
        fs = 256000
        seconds = 1

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press Enter to record next or to stop press ctrl + F2 ({i + 1}/{n_times}): ")


def record_background_sound(save_path, n_times=50):
    """
    This function will run automatically `n_times` and record your background sounds so you can make some
    keybaord typing sound and saying something gibberish.
    Note: Keep in mind that you DON'T have to say the wake word this time.
    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.
    save_path: str
        Where to save the wav file which is generated in every iteration.
        Note: DON'T set it to the same directory where you have saved the wake word or it will overwrite the files.
    """

    input("To start recording your background sounds press Enter: ")
    for i in range(n_times):
        fs = 256000
        seconds = 1

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        print(f"Currently on {i + 1}/{n_times}")


# Step 1: Record yourself saying the Wake Word (go)
print("Recording the Wake Word:\n")
record_audio_and_save("all_data/go_data/", n_times=50)

# Step 2: Record your background sounds (Just let it run, it will automatically record)
print("Recording the Background sounds:\n")
record_background_sound("all_data/background_sound/", n_times=100)

###############################################################################################
