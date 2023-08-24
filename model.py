n_mfcc = 13 
max_length = 700 

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
