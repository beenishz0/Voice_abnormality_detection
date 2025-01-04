import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the sampling rate and maximum length for audio files
SAMPLE_RATE = 16000  # Typical sample rate for speech
MAX_LENGTH = 1680000   # 1.45 minute (16000*60) as 1 second of audio at SAMPLE_RATE is 16000 (you can adjust this based on your dataset). For rainnow passage most audio's were around 1.45 minutes hence 168000 

# 1. Function to load and preprocess the .wav files into raw waveforms
def load_audio_files(directory, max_length=MAX_LENGTH):
    audio_data = []
    labels = []

    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                if file_name.endswith('.wav'):
                    # Load audio file
                    audio_path = os.path.join(label_path, file_name)
                    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

                    # Padding or truncating to ensure fixed length
                    if len(audio) > max_length:
                        audio = audio[:max_length]
                    elif len(audio) < max_length:
                        audio = np.pad(audio, (0, max_length - len(audio)))

                    audio_data.append(audio)
                    labels.append(0 if label == 'normal' else 1)  # Label: 0 = normal, 1 = disease

    return np.array(audio_data), np.array(labels)
# 2. Load your data
directory = '/path/to/dataset'  # Provide path to your dataset containing 'normal' and 'disease' folders
X, y = load_audio_files(directory)

# 3. Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build the model using CNN (Convolutional Neural Network)
def build_cnn_model(input_shape):
    model = models.Sequential()

    # 1st convolutional block
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))

    # 2nd convolutional block
    model.add(layers.Conv1D(64, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # 3rd convolutional block
    model.add(layers.Conv1D(128, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Flatten the features
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))

    # Output layer: binary classification
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# 5. Compile the model
model = build_cnn_model((MAX_LENGTH, 1))  # Raw waveform input shape
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Print the model summary
model.summary()

# 7. Reshape data for CNN input
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (samples, time, 1)
X_test = np.expand_dims(X_test, axis=-1)

# 8. Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 9. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# 10. Make predictions
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

# 11. Example of visualizing the accuracy during training
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy Over Epochs')
plt.show()

# 12. Save the model --> here model is saved as voice_abnormality_detection_model.keras
model.save("voice_abnormality_detection_model.keras")

