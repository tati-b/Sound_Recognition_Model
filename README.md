# Sound Recognition Model
This model can identify different types of sounds, such as dripping water, flowing water, and silence, from audio recordings.

## Steps Before Training the Model
* __Recorded Audio Data__:
Audio samples representing various sound categories, such as dripping water, flowing water, and silence, were recorded in the past.
* __Preprocessed Audio Data__:
The recorded audio samples were previously trimmed to a consistent length, approximately 5 to 6 seconds, using audio editing tools.
The audio recordings were converted to the WAV format during previous preprocessing.
* __Organized Data Folders__:
The audio samples were sorted into distinct folders for each sound category, such as "dripping_water," "flowing_water," and "silence."
These categorized folders were placed within the main "training," "validation," and "testing" folders.

## Train the Sound Recognition Model
* __Run train_model.py__:
Open your terminal or command prompt.
Navigate to the directory containing the train_model.py file using the cd command.
Execute the script using the command: python train_model.py

The script will:
1. Extract Mel-frequency cepstral coefficients (MFCCs) from the audio data and perform necessary preprocessing.
2. Build, compile, and train a sound recognition model using the processed data.
3. Evaluate the model's performance on testing data and display accuracy metrics for each sound category.

## Results
After training the model and evaluating its performance, the following accuracy metrics were obtained:

* Overall accuracy: 0.67 (approximately)
* Dripping accuracy: 0.67
* Flowing accuracy: 1.00
* Silence accuracy: 0.33
  
These accuracy values provide insights into how well the model can recognize different sound categories based on the provided audio data.

## Model Architecture
The model architecture is designed to learn and classify different sound categories based on the extracted MFCCs from audio samples.

```
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
```
__Convolutional Layer (Conv2D)__:

* Purpose: This layer performs convolution operations on the input data (MFCCs) using a set of learnable filters. It helps detect simple features and patterns in the audio data.
* How: Each filter scans a small region of the input data and extracts features. The ReLU (Rectified Linear Unit) activation function is applied to introduce non-linearity.
* Utilizes 32 filters of size 3x3 to detect fundamental patterns in MFCCs.
  
__Max Pooling Layer (MaxPooling2D)__:
* Purpose: Max pooling reduces the dimensions of the feature maps, focusing on the most important information and discarding less significant details.
* How: It divides the input into non-overlapping regions and selects the maximum value from each region, reducing the size of the data while retaining the most relevant features.
* Downsamples the data by selecting maximum values from a 2x2 region.
  
__Convolutional Layer (Conv2D)__:
* Purpose: Building on the initial features, this layer employs more filters to capture higher-level patterns and relationships.
* How: It performs convolutions with a new set of filters, further enhancing the model's ability to recognize complex features.
* Employs 64 filters of size 3x3 to capture more complex patterns.
  
__Max Pooling Layer (MaxPooling2D)__:
* Purpose: Similar to the previous max pooling layer, this one continues to downsample the feature maps, enhancing the abstraction of features.
* How: It reduces the spatial dimensions of the data while preserving the most important features.

__Flatten Layer__:
* Purpose: Converts the 2D feature maps into a 1D vector, preparing the data for input into the fully connected layers.
* How: It reshapes the data while maintaining the relationships between features, allowing the fully connected layers to process them.
  
__Dense Layer (Fully Connected)__:
* Purpose: These layers process complex combinations of features learned from the convolutional layers.
* How: The dense layers apply linear transformations followed by the ReLU activation function to analyze and transform the features.
  
__Dropout Layer__:

* Purpose: Dropout helps prevent overfitting by randomly deactivating a specified fraction of neurons during training.
* How: By randomly removing neurons, the model avoids relying heavily on specific features and learns a more generalized representation.
  
__Dense Layer (Fully Connected)__:

* Purpose: This additional dense layer further refines the learned feature combinations.
* How: Similar to the previous dense layer, it performs linear transformations and ReLU activations to capture intricate patterns.
  
__Output Layer (Dense)__:

* Purpose: The final layer produces class probabilities for each sound category, enabling classification.
* How: The softmax activation function converts the final layer's outputs into probability distributions over the categories, determining the most likely sound category for a given input.
* Concludes the architecture with 3 units (sound categories).

## Trained Model
The trained model's architecture and implementation can be found in the `audio_model/audio_model.h5` file. The size of the model file is approximately 16.5 MB.

### Evaluation

To evaluate the model's performance on your own testing data, you can use the provided `run_model.py` script. This script loads the model and performs inference on the testing data, providing accuracy as an evaluation metric.

### Requirements

The model has been trained using Python with the following major libraries:
- Keras (part of TensorFlow) for model creation and training
- Librosa for audio feature extraction
- NumPy for array manipulation

### Usage

1. Ensure you have the necessary libraries installed (Keras, Librosa, NumPy).
2. Run the `run_model.py` script to load the trained model and evaluate its accuracy on your testing data.

## Results
After running the trained model and evaluating its performance, the following accuracy metric was obtained:
* Restored model accuracy: 0.67

**Note**: The testing data used for model evaluation is the same data used during the initial training and testing process. This was done to show the model's functionality and provide a direct comparison to the initial training accuracy (0.67). For accurate and unbiased model evaluation, it's recommended to employ distinct and unseen testing data.
