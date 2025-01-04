import os
import numpy as np
import librosa
import tensorflow as tf

# Define constants
SAMPLE_RATE = 16000  # The sample rate used during training
MAX_LENGTH = 1680000   # The length of audio (105 sec (1min 45) second) used during training. this should match the max_length used in training of the model 

# Function to load and preprocess audio
def preprocess_audio(audio_path, max_length=MAX_LENGTH):
    # Load the audio file with librosa
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Pad or truncate the audio to the fixed length
    if len(audio) > max_length:
        audio = audio[:max_length]
    elif len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))

    # Reshape to add channel dimension for CNN input (samples, time, 1)
    return np.expand_dims(audio, axis=-1)

# Function to preprocess multiple audio files in a directory
def preprocess_batch(directory, max_length=MAX_LENGTH):
    audio_data = []
    file_paths = []

    # Iterate through the files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):  # Process only .wav files
            file_path = os.path.join(directory, file_name)
            processed_audio = preprocess_audio(file_path, max_length)
            audio_data.append(processed_audio)
            file_paths.append(file_path)

    return np.array(audio_data), file_paths

# Load the pre-trained model
model = tf.keras.models.load_model('path/to/saved/model/voice_abnormality_detection_model.keras')  # Path to your saved model

# Path to the directory containing the new audio files
new_data_directory = '/path/to/new/dataset/on/which/predication/needs/to/happen'  # Update with the correct path

# Preprocess the batch of new audio files
X_new, file_paths = preprocess_batch(new_data_directory)

# Make predictions for the batch
predictions = model.predict(X_new)

# Convert predictions to labels (normal or diseased)
prediction_labels = ['normal' if p < 0.5 else 'diseased' for p in predictions]   ### here the model predicts if the audio file is normal or diseased

# Print the results for each file
for file_path, label in zip(file_paths, prediction_labels):
    print(f"{file_path}: {label}")
