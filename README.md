# The Empathy Engine 🎙️

## Vision
Make AI voices sound more human, adding empathy and emotion into TTS.
## Description:
The Empathy Engine is an AI-powered service that transforms text into expressive, human-like speech by detecting the emotion in the input text and modulating vocal parameters such as pitch, rate, and volume accordingly. Unlike standard TTS systems that sound robotic, this project enables the AI to sound enthusiastic for positive messages, calm and patient for frustrated messages, and neutral for ordinary statements. The system integrates emotion detection using NLP models and advanced text-to-speech synthesis, producing audio output that conveys emotion and enhances user engagement.

## Key Features:

Accepts text input via a web interface.

Detects and classifies emotion into multiple categories (Positive, Negative, Neutral).

Modulates voice parameters (pitch, speed, and volume) based on detected emotion.

Generates playable audio output (.mp3/.wav) in real-time.

Optional enhancements include granular emotion detection, intensity scaling, and an interactive web UI.

## Impact:
By bridging the gap between text and expressive speech, The Empathy Engine makes AI interactions feel more natural, trustworthy, and emotionally intelligent, improving engagement in customer service, sales, and conversational AI applications.

## Features
- Emotion detection (joy, anger, sadness, neutral).
- Modulated speech output (pitch, rate, volume).
- Flask web app with text input and audio output.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run app: `python app.py`
3. Open: `http://127.0.0.1:5000`


