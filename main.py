# ###### IMPORTS #############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
# #### Loading saved csv ##############
df = pd.read_csv("final_audio_data_csv/audio_data1.csv")
df.info()

# ###### Making our data training-ready
# We will extract data from CSV into a 2D array "X" (our dataset)

Samples = 10   # Each 1s audio Clip is sampled at 0.1s windows during preprocessing
X = []
for i in range(1, 21):
    X.append(df[F"filter bank {i}"].tolist())
for i in range(1, 21):
    X.append(df[F"1st derivative {i}"].tolist())
for i in range(1, 21):
    X.append(df[F"2nd derivative {i}"].tolist())

X = np.array(X).T
print(X.shape)
X = np.reshape(X, (int(len(X)/Samples), Samples, 60))  # Final shape for X (data instances, samples, features)
print(X.shape)
# The labeled classes of the audio files: 1 is for WUW word found. (WUW is "go")
y = np.array(df["labels"].tolist())
y = np.reshape(y, (int(len(y)/Samples), Samples))
y = np.reshape(y[:, 1], (len(y), 1))                # Final shape for X (data instances, features)
print(y.shape)


# ###### train test split ############
# Here, we split the data into training and testing data.
# No data ratio was provided by the paper, so we will use 80% for training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
dfy = pd.DataFrame(y_train, columns=['label1'])
print(dfy)
print(X_train.shape)
print(y_train.shape)

# building the model

# Add 5 bidirectional LSTMs based on the paper architecture
model = keras.Sequential()
model.add(Input((Samples, 60)))
model.add(BatchNormalization())  # This layer is added to improve accuracy. It is not found in paper model
model.add(Bidirectional(LSTM(320, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(320, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(320, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(320, dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(320, dropout=0.2,)))

# Adding a dense layer that represents the different possible characters to be
# identified (A->Z ,.?!;). This layer is the output of the paper and loss is computed by
# ctc loss. In this representation I built, a second dense layer is added to specify
# whether the WUW is detected or not and loss is of binary cross-entropy.

model.add(Dense(30, kernel_initializer='normal', activation='relu'))  # kernel_initializer='normal'

# Add a classifier output layer (to classify WUW detected or not)
model.add(Dense(1, activation='sigmoid'))

# Compiling model and starting training

opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()
# Use early stopping to reach best result when accuracy plateaus as epochs number is higher
es = EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=3, min_delta=0.01)
history = model.fit(X_train, y_train, batch_size=25, epochs=100, validation_split=0.1, callbacks=[es])

# Visualising training and validation performance (plotting accuracy and loss functions)
plt.subplot(1, 2, 1)  # plot 1: Training and Validation Accuracies
plt.plot(history.history['binary_accuracy'], color='g')
plt.title("Training Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['val_binary_accuracy'], color='b')
plt.title("Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(["Training", "Validation"], loc="lower right")
plt.subplot(1, 2, 2)  # plot 2: Training and Validation Loss Functions
plt.plot(history.history['loss'], color='g')
plt.title("Training Loss Function")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['val_loss'], color='b')
plt.title("Validation Loss Function")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(["Training", "Validation"])
plt.show()

# Evaluating model on test data
print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Saving model to be used later on real-time usage
model.save("saved_model/WWD.h5")


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for some samples")
predictions = model.predict(X_test[:20])
exact = y_test[:20]
for i in range(len(predictions)):
    print("predictions:", predictions[i], "exact value: ", exact[i])

# ### Evaluating our model ###########

print("\n Model Classification Report: \n")
# Predicting all test set
y_pred = model.predict(X_test)
for i in range(len(y_pred)):
    if y_pred[i] > 0.95:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# Making the Confusion Matrix
cm_labels = ["No Wakeup Word", "Wakeup Word said"]
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
disp.plot()

plt.show()
