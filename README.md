# Speech Emotion Recognition (SER) App

Ram AZZAM -
Emile SASSINE - 
Mohamed ELADEB - 
Khaled ABOU KHACHFE




A lightweight application and training pipeline to classify human emotions from audio. It compares two different Deep Learning models trained on the RAVDESS dataset and includes a GUI to test them live with your microphone.

## Overview

- **The Data:** Trained on the RAVDESS dataset (8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised).
- **The Pipeline:** Audio is preprocessed into 3-second, 16 kHz mono clips, then converted into 64-band log-Mel spectrograms. The dataset was split strictly by **actor** (speaker-independent) to prevent data leakage.
- **Model 1 (CNN from scratch):** A custom 4-block Convolutional Neural Network trained entirely from scratch.
- **Model 2 (ResNet-18):** Uses transfer learning from an ImageNet pre-trained ResNet-18, with a frozen backbone and a newly trained linear classifier.

## Folder Structure

Ensure your project is structured like this before running the application:

```text
Project_Folder/
 ├── code/
 │    └── app.py
 ├── model1/
 │    ├── best_model.pt
 │    └── normalization_stats.json
 └── model2/
      ├── best_model.pt
      └── normalization_stats.json
```

## How to Run

You don't need to manually install dependencies beforehand. The script is autonomous and will automatically install missing libraries (like `torch`, `librosa`, `sounddevice`, etc.) on its first run.

1. Navigate to your project folder.
2. Run the application
3. A GUI will open. Click **"Enregistrer" (Record)**, speak for 3 seconds, and the app will display the log-Mel spectrogram of your voice alongside the probability predictions from both models. (You might need to wait a couples seconds..)

## Limitations
- The models were trained on acted speech from native English speakers in a studio environment. 
- Live predictions might vary based on your microphone quality, background noise, language, and recording volume.