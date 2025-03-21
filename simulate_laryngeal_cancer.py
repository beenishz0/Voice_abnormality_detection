## This is a script that takes in voice waveform files from a folder and modulates it to sound like a voice with laryngeal cancer 

import os
import random
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup, normalize
from scipy.signal import sawtooth
import librosa
import librosa.display
import matplotlib.pyplot as plt


# Function to apply random pitch manipulation (hoarseness or breathiness simulation)
def manipulate_pitch(audio, min_factor=0.85, max_factor=1.2):
    """
    Randomly adjust the pitch of the audio to simulate hoarseness or irregular voice.
    """
    pitch_factor = random.uniform(min_factor, max_factor)
    samples = np.array(audio.get_array_of_samples())
    # Create a numpy array for the audio
    manipulated_samples = librosa.effects.pitch_shift(samples.astype(float), sr=audio.frame_rate, n_steps=pitch_factor)
    manipulated_audio = AudioSegment(
        manipulated_samples.astype(np.int16).tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=audio.channels,
    )
    return manipulated_audio


# Function to manipulate speed (altering tempo or making the voice slow and labored)
def manipulate_speed(audio, min_speed=0.75, max_speed=1.25):
    """
    Randomly change the speed of the audio to simulate slow or erratic speech due to laryngeal cancer.
    """
    speed_factor = random.uniform(min_speed, max_speed)
    manipulated_audio = speedup(audio, playback_speed=speed_factor)
    return manipulated_audio


# Function to add random background noise (mimicking weakness or breathiness)
def add_noise(audio, noise_level=0.02):
    """
    Add random noise to the audio, simulating a breathy or weak voice.
    """
    samples = np.array(audio.get_array_of_samples())
    noise = np.random.normal(0, noise_level, samples.shape[0])
    noisy_samples = samples + noise.astype(np.int16)
    noisy_audio = AudioSegment(
        noisy_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=audio.channels,
    )
 return noisy_audio


# Function to apply distortion (for irregular, trembling sounds)
def apply_distortion(audio, distortion_factor=0.5):
    """
    Randomly apply distortion to the audio to simulate trembling or irregular vocal folds.
    """
    samples = np.array(audio.get_array_of_samples())
    distorted_samples = samples * distortion_factor
    distorted_samples = np.clip(distorted_samples, -32768, 32767)  # Ensure 16-bit sample range
    distorted_audio = AudioSegment(
        distorted_samples.astype(np.int16).tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=audio.channels,
    )
    return distorted_audio


# Function to manipulate the volume (hoarse or low-volume voice)
def manipulate_volume(audio, min_volume=-5, max_volume=0):
    """
    Randomly adjust the volume of the audio to simulate weakness or hoarseness.
    """
    volume_change = random.uniform(min_volume, max_volume)
    manipulated_audio = audio + volume_change
    return manipulated_audio


# Main function to process a directory of .wav files and create manipulated versions
def process_audio_files(input_dir, output_dir, num_samples=266):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    if len(audio_files) < num_samples:
        print(f"Warning: There are less than {num_samples} audio files. Using {len(audio_files)} files.")
        num_samples = len(audio_files)

    for i, file_name in enumerate(audio_files[:num_samples]):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing {file_path}...")

        # Load the audio file
        audio = AudioSegment.from_wav(file_path)

        # Apply random manipulations
        manipulated_audio = audio
        manipulations = random.randint(1, 5)

        for _ in range(manipulations):
            manipulation_type = random.choice(['pitch', 'speed', 'noise', 'distortion', 'volume'])
            if manipulation_type == 'pitch':
                manipulated_audio = manipulate_pitch(manipulated_audio)
            elif manipulation_type == 'speed':
                manipulated_audio = manipulate_speed(manipulated_audio)
            elif manipulation_type == 'noise':
                manipulated_audio = add_noise(manipulated_audio)
            elif manipulation_type == 'distortion':
                manipulated_audio = apply_distortion(manipulated_audio)
            elif manipulation_type == 'volume':
                manipulated_audio = manipulate_volume(manipulated_audio)

        # Normalize to ensure consistent volume
        manipulated_audio = normalize(manipulated_audio)

        # Save the manipulated audio file
        output_file_name = f"simulated_{i + 1}_{file_name}"
        output_file_path = os.path.join(output_dir, output_file_name)
        manipulated_audio.export(output_file_path, format="wav")
        print(f"Saved manipulated file: {output_file_path}")


if __name__ == "__main__":
    # Define the input and output directories
    input_directory = "path_to_input_files_directory"  # Replace with your input folder path
    output_directory = "New_output_wav_files"  # Replace with your output folder path

    # Process the audio files
    process_audio_files(input_directory, output_directory)
