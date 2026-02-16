# Speech Emotion Recognition using CNN

A deep learning project that recognizes human emotions from speech audio files using Convolutional Neural Networks (CNN). The system classifies audio into five emotion categories: Calm, Happy, Sad, Angry, and Fearful.

## Project Overview

This project implements a Speech Emotion Recognition (SER) system that:
- Extracts MFCC (Mel-frequency Cepstral Coefficients) features from audio files
- Uses a 1D Convolutional Neural Network for emotion classification
- Provides a web interface for real-time emotion prediction
- Trains on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset

## Features

- **5 Emotion Classification**: Calm, Happy, Sad, Angry, Fearful
- **MFCC Feature Extraction**: Uses 13 MFCC coefficients for audio analysis
- **Data Augmentation**: Implements noise addition and pitch shifting to improve model robustness
- **Web Interface**: Flask-based web application for easy audio file upload and emotion prediction
- **CNN Architecture**: 1D Convolutional Neural Network with multiple layers for feature learning

## Project Structure

```
SpeechEmotionRecognition_CNN/
├── app.py                 # Flask application configuration
├── cnn.py                 # CNN model training and prediction functions
├── main (1).py            # Flask routes and web interface logic
├── templates/
│   └── upload2.html       # Web interface template
├── static/                # Static files (images, audio samples)
├── data/                  # RAVDESS dataset directory
├── model/                 # Saved model weights and architecture
│   └── aug_noiseNshift_2class2_np.h5
├── model.json             # Saved model architecture
└── README.md
```

## Requirements

### Python Packages

- **Flask** - Web framework
- **Flask-Bootstrap** - Bootstrap integration for Flask
- **TensorFlow** - Deep learning framework
- **Keras** - High-level neural networks API
- **librosa** - Audio and music analysis library
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Plotting library
- **seaborn** - Statistical data visualization
- **scipy** - Scientific computing
- **tqdm** - Progress bars

## Dataset

The project uses the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The dataset should be organized in a `data/` directory with subdirectories containing audio files.

### Dataset Format
- Audio files are expected to be in WAV format
- File naming convention: `[modality]-[vocal_channel]-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav`
- Emotions mapped: 2=Calm, 3=Happy, 4=Sad, 5=Angry, 6=Fearful

## Model Architecture

The CNN model consists of:
- **Input Layer**: MFCC features (13 coefficients)
- **Convolutional Layers**: Multiple 1D Conv layers with 256, 128, and 64 filters
- **Activation**: ReLU activation functions
- **Regularization**: Batch Normalization and Dropout (0.25)
- **Pooling**: MaxPooling1D layers
- **Output Layer**: Dense layer with 5 units (softmax activation for 5 emotion classes)

### Training Parameters
- **Optimizer**: SGD (Stochastic Gradient Descent) with learning rate 0.0001
- **Loss Function**: Categorical cross-entropy
- **Batch Size**: 16
- **Epochs**: 700
- **Validation Split**: 20% (Stratified Shuffle Split)
- **Callbacks**: ModelCheckpoint, ReduceLROnPlateau

## Usage

### Training the Model

1. Ensure the RAVDESS dataset is placed in the `data/` directory
2. Run the training script:

```bash
python cnn.py
```

The script will:
- Load and preprocess audio files from the `data/` directory
- Extract MFCC features from each audio file
- Apply data augmentation (noise and pitch shifting)
- Train the CNN model
- Save the model architecture to `model.json` and weights to `model/aug_noiseNshift_2class2_np.h5`
- Generate evaluation metrics and confusion matrix

### Running the Web Application

1. Ensure the trained model files exist (`model.json` and `model/aug_noiseNshift_2class2_np.h5`)
2. Start the Flask application:

```bash
python "main (1).py"
```

3. Open your browser and navigate to `http://localhost:5000`
4. Upload a WAV audio file through the web interface
5. The predicted emotion will be displayed on the page

### Prediction Function

The `web_test_label()` function in `cnn.py` can be used programmatically:

```python
import cnn

emotion = cnn.web_test_label("path/to/audio.wav")
print(f"Predicted emotion: {emotion}")
```

## Data Augmentation

The project implements two augmentation methods to reduce overfitting:

1. **Noise Addition**: Adds white noise to audio signals
2. **Pitch Shifting**: Modifies the pitch of audio signals

These augmentations are applied during training to increase dataset size and improve model generalization.

## File Processing

- **Audio Duration**: 3 seconds (configurable via `input_duration` variable)
- **Sample Rate**: 44100 Hz (22050 * 2)
- **Feature Extraction**: 13 MFCC coefficients averaged across time frames
- **Offset**: 0.5 seconds (to skip initial silence)

## Model Output

The model predicts one of five emotions:
- **CALM** (class 0)
- **HAPPY** (class 1)
- **SAD** (class 2)
- **ANGRY** (class 3)
- **FEARFUL** (class 4)

## Limitations

As noted in the project documentation:

1. The model is trained on English language audio only
2. Currently optimized for male voices (can be modified for female voices)
3. May struggle with fake or acted emotions
4. Performance may vary with different accents and geographical variations
5. Requires a quiet environment for optimal accuracy

## Applications

Speech Emotion Recognition can be used in various domains:

- **Medicine**: Analyzing patients' emotional states during counseling
- **E-learning**: Adapting presentation style based on learner's emotional state
- **Call Centers**: Prioritizing calls based on caller's emotional tone
- **Smart Home Assistants**: Responding based on user's mood
- **Security Systems**: Detecting suspicious emotional patterns

## Key Learnings

- CNN proved to be the most effective algorithm for this speech recognition task
- Data augmentation (noise and pitch shifting) helps combat overfitting
- Model accuracy decreases as the number of emotion classes increases
- Dataset quality directly impacts model performance
- Pre-processing and data manipulation are crucial steps in the ML pipeline

## References

- RAVDESS Dataset: [Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
- Librosa Documentation: [librosa.github.io](https://librosa.github.io/librosa/)
- TensorFlow/Keras: [tensorflow.org](https://www.tensorflow.org/guide/keras)
- Scikit-learn: [scikit-learn.org](https://scikit-learn.org/)

## Author

**Yavar Khan**

Bachelor of Technology in Information Technology  
Amity University Uttar Pradesh, Noida

## License

This project was developed as part of academic coursework.
