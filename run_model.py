import numpy as np
from keras.models import load_model

# Load the saved model
new_model = load_model('audio_model/audio_model.h5')

# Load testing data from the saved numpy array
testing_data = np.load('testing_data.npy')
testing_labels = np.load('testing_labels.npy') 

# Evaluate the loaded model on testing data
test_loss, test_accuracy = new_model.evaluate(testing_data[..., np.newaxis], testing_labels)
print("Restored model accuracy:", test_accuracy)