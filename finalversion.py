import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator
import librosa

training_folder = "D:\\project\\samples\\training"
validation_folder = "D:\\project\\samples\\validation"
testing_folder = "D:\\project\\samples\\testing"

n_mfcc = 13 # Number of coefficients to extract
max_length = 700 # 700/100=7sec 

batch_size = 5
epochs = 50


def extract_and_pad(audio_path, n_mfcc, max_length):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max(max_length - mfccs.shape[1], 0)
    padded_mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')[:, :max_length]
    return padded_mfccs

def generate_data_and_labels(data_dir, n_mfcc, max_length):
    audio_paths = []
    labels = []
    for label, class_name in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                audio_path = os.path.join(class_dir, filename)
                audio_paths.append(audio_path)
                labels.append(label)
    num_samples = len(audio_paths)
    data = np.zeros((num_samples, n_mfcc, max_length)) 
    for i, audio_path in enumerate(audio_paths):
        mfccs = extract_and_pad(audio_path, n_mfcc=n_mfcc, max_length=max_length)
        data[i] = mfccs
    labels = to_categorical(labels, num_classes=3) 
    return data, labels

# Generate MFCC data and corresponding labels for each dataset
training_data, training_labels = generate_data_and_labels(training_folder, n_mfcc, max_length)
validation_data, validation_labels = generate_data_and_labels(validation_folder, n_mfcc, max_length)
testing_data, testing_labels = generate_data_and_labels(testing_folder, n_mfcc, max_length)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(n_mfcc, max_length, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

#sparse_categorical_crossentropy?   categorical_crossentropy - produces a one-hot array containing the probable match for each category?
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_data[..., np.newaxis], training_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data[..., np.newaxis], validation_labels))

# Evaluate model's performance on the testing data
test_loss, test_accuracy = model.evaluate(testing_data[..., np.newaxis], testing_labels)
print("Overall accuracy:", test_accuracy)

predictions = model.predict(testing_data[..., np.newaxis])
predicted_labels = np.argmax(predictions, axis=1)

dripping_indices = np.where(testing_labels[:, 0] == 1)[0]
dripping_accuracy = np.mean(predicted_labels[dripping_indices] == 0)
print("Dripping accuracy:", dripping_accuracy)

flowing_indices = np.where(testing_labels[:, 1] == 1)[0]
flowing_accuracy = np.mean(predicted_labels[flowing_indices] == 1)
print("Flowing accuracy:", flowing_accuracy)

silence_indices = np.where(testing_labels[:, 2] == 1)[0]
silence_accuracy = np.mean(predicted_labels[silence_indices] == 2)
print("Silence accuracy:", silence_accuracy)


