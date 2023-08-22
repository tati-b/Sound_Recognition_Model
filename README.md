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

