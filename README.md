# Voice_abnormality_detection
This repository provides a script to generate an AI model that can be trained on voice dataset for predicting voice abnormalities like laryngeal cancer. 
Additionally, the repository includes a prediction script to test the trained model on new datasets, as well as scripts on how to generate synthesized voice data from text and simulate cancer voice. 

The model generated uses CNN to get features from raw data instead of using MFCC.  

It gave a prediction accuracy of ~72-88% on training data of ~180 samples and eval data of ~24 samples running on a 2S server CPU system with Ubuntu OS. 

# Explantion on The Model 

This AI model was created in TensorFlow that takes raw audio files (in .wav format), processes them using Convolutional Neural Networks (CNNs) and classifies the voice as either "normal" or from a person with a disease. 

1) Loading Audio Files: 

  The load_audio_files function loads .wav files from a directory. The audio files are loaded using librosa.load() with a specified SAMPLE_RATE. 
  
  Audio files are padded or truncated to the MAX_LENGTH to ensure consistency in input size. 

 2) Preprocessing: 

  Raw waveforms are directly used as features. We reshape the data to make it compatible with CNNs by adding a channel dimension (1 for mono audio). 

3) Model Architecture: 

The model is a 1D Convolutional Neural Network (CNN), which is well-suited for analyzing time-series data like audio waveforms. 

The model consists of three convolutional layers followed by max-pooling and fully connected layers. The output is a single neuron with a sigmoid activation function for binary classification (normal or disease). 

4) Training: 

 The model is compiled using the Adam optimizer and binary cross-entropy loss function since it is a binary classification problem. 

 The model is trained for 10 epochs with a batch size of 32. 

5) Model Evaluation: 

  After training, the model is evaluated on the test set, and accuracy is printed. 

6) Visualization (optional): 

  A simple plot visualizes the training and validation accuracy over epochs to check if the model is overfitting or underfitting. 

Important Notes: 

Data Preprocessing: Ensure that your audio data is consistent in terms of length. Padding/truncating to a fixed length (e.g., 1 second of audio) is necessary for feeding the raw waveforms into the neural network. 

Model Choice: The CNN model works well with raw audio waveforms because it can learn spatial hierarchies in the data. For more complex time dependencies, you could also try Recurrent Neural Networks (RNNs) like LSTMs or GRUs. 

Dataset: Ensure you have a balanced dataset with sufficient labeled samples for training. The labels should be clearly defined as "normal" and "disease." 

This is a basic implementation, and you can further improve the model by adding data augmentation, regularization, hyperparameter tuning, or exploring more advanced architectures like 1D CNN + RNN hybrid models. 

Troubleshooting and Tips 

  Audio File Format: Ensure that all audio files are in .wav format. If your files are in a different format (e.g., .mp3), use a tool like ffmpeg or librosa to convert them to .wav files. 
  
  Memory Issues: If you're working with a large dataset, you might run into memory issues. In this case, you can: 
  
  Use a smaller dataset for testing. 
  
  Reduce the length of the audio or perform downsampling. 
  
  Use ImageDataGenerator or similar techniques to load and preprocess data in batches rather than all at once. 
  
  Model Performance: If the model doesn't perform well, consider tuning hyperparameters, increasing the model's complexity, or using data augmentation techniques (e.g., pitch shifting, time stretching) to improve the generalization. 
  
  Feature Engineering: If you decide to experiment with other features (like MFCCs), you can modify the code to extract those features instead of using raw waveforms. However, this code assumes you're working with raw audio for automatic feature extraction. 

# How To Run The Script To Generate a Trained Model

To run the python script (train_audio_model.py) to generate a trained model for detecting larynx disease in voice data, you'll need to follow a few key steps to set up your environment, prepare the data, and execute the code. 
Here's a guide to help you run the code effectively: 

1. Set Up Your Environment 

Make sure you have Python installed on your system, along with all the necessary libraries and dependencies. If you don't have Python installed, download and install it from here. 

1.1 Create a Virtual Environment (Optional but Recommended) 

This helps isolate dependencies for your project: 

On Linux/macOS 

python3 -m venv larynx-env 

Activate the virtual environment 

On Linux/macOS 

source larynx-env/bin/activate 

1.2 Install Required Libraries 

  Install the required libraries using pip. The libraries needed for the code include: 
  
  TensorFlow 
  
  Librosa (for audio processing) 
  
  Matplotlib (for plotting) 
  
  Scikit-learn (for splitting the dataset) 
  
  NumPy 
  
  Run the following commands to install these dependencies: 
  
  pip install tensorflow librosa matplotlib scikit-learn numpy 

2. Prepare the Dataset 

  You need a labeled dataset of normal and diseased voice samples. The code assumes you have two folders containing audio files (.wav format): 
  
  Normal voice recordings: Store these in the folder path_to_normal_audio_files/. 
  
  Diseased voice recordings: Store these in the folder path_to_diseased_audio_files/. 
  
  Make sure the audio files are in WAV format, as the code uses librosa.load() to read .wav files. 
  
  You can create these directories like this: 
  
  mkdir path_to_normal_audio_files 
  
  mkdir path_to_diseased_audio_files 

3. Run the Code 

Now, let's break down the steps to run the code. 

3.1 Prepare Your Python Script 

Use the prediction Python script provided in this repository.  

3.2 Modify the Dataset Paths 

  Make sure to change the paths for NORMAL_AUDIO_PATH and DISEASED_AUDIO_PATH to point to the actual locations of your audio data. 
  
  Modify these paths to where your audio files are located 
  
  NORMAL_AUDIO_PATH = 'path_to_normal_audio_files/*.wav' 
  
  DISEASED_AUDIO_PATH = 'path_to_diseased_audio_files/*.wav' 
  
   
  If you're unsure about your audio files' format, you can check with librosa by printing the y array from librosa.load() to inspect the audio data. 

3.3 Execute the Script 

Once you've set up the paths and ensured that your environment is properly configured, you can run the script. 

In your terminal or command prompt, navigate to the folder where the Python script is located and run it using the following command: 

python3 train_audio_model.py 

4. Monitor the Progress 

During execution, the script will: 

Load the audio data from the specified folders. 

Extract features (Mel-spectrograms) from each audio file. 

Train the model (the CNN model) on the training data. 

Evaluate the model on a test set and print the results (accuracy). 

Plot training history (accuracy and loss over epochs). 

If the script completes successfully, you'll see a message displaying the test accuracy of the model, and graphs showing training/validation accuracy and loss. 

5. Use the Model for Predictions 

After the model has been trained, you can use it to make predictions on new voice samples. To test the model with a new voice file, you can use the provided prediction_script_for_newdata.pay to test the trained model on new datasets. 

Replace the paths in the script with the actual file path of a new .wav audio file(s). 

6. Save and Load the Model 

After training, the model is saved as larynx_disease_model.h5 using model.save(). To load the trained model later for predictions, you can use: 

Load the saved model 

model = tf.keras.models.load_model('voice_abnormality_detection_model.keras') 

7. Improvement and Adjustments 

Hyperparameter Tuning: If you find that the model doesn't perform well, you may want to experiment with tuning the hyperparameters, such as the number of layers, the number of filters in each convolutional layer, or the number of epochs. 

Data Augmentation: You can augment your audio dataset by adding noise, changing the pitch, or varying the speed of the recordings to make your model more robust. 

More Advanced Models: If a simple CNN doesn't give satisfactory results, consider using more advanced architectures like LSTMs, GRUs, or even pre-trained models for audio classification (like VGGish or YAMNet). 

Troubleshooting 

If you encounter issues, here are some common errors and solutions: 

FileNotFoundError: If your paths to the audio files are incorrect, ensure that the paths you provide for NORMAL_AUDIO_PATH and DISEASED_AUDIO_PATH are correct. 

TensorFlow or Keras Errors: Make sure you're using compatible versions of TensorFlow and Keras. You can check the TensorFlow version with pip show tensorflow. 

Shape Mismatch: If you get an error about input shapes, ensure that all audio features are resized to the same shape before feeding them into the model. 


By following these steps, you'll have a working environment to train and test the model for classifying voices as either normal or from a person with larynx disease. You can then further improve the model by fine-tuning the parameters or using more advanced neural network architectures. 

# How to Use the Prediction Script 

Save the Trained Model: After training your model, make sure to save it using the following code (already included in the training script): 

model.save("voice_abnormality_detection_model.keras") 
  

Ensure that the saved model file is in the correct directory or update MODEL_PATH accordingly. 

Prepare the Input Directory: Place the .wav files you want to classify into a directory (e.g., test_audio_files). Update the INPUT_AUDIO_DIR variable in the inference script to point to this directory. 

Run the Script: You can now run the predict_script_for_newdata.py script to classify the audio files. 

python predict_script_for_newdata.py 
  

Output: The script will print predictions for each .wav file in the input directory: 

Processing voice_001.wav... 
Prediction for voice_001.wav: Normal 
 
Processing voice_002.wav... 
Prediction for voice_002.wav: Diseased 
  

Notes: 

Ensure the path to your audio files and the model file is correct. 

The script assumes the model was trained for binary classification (normal or diseased), using sigmoid activation and binary_crossentropy loss. 

The .wav files should be preprocessed in the same way as the training data (e.g., same sample rate, same number of MFCC features). 

This script will take in any number of .wav files, preprocess them, and use the trained model to classify them as either normal or diseased based on the voice characteristics. 

