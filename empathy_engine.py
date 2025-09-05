import os
import nltk
from transformers import pipeline
from gtts import gTTS
from pydub import AudioSegment
import numpy as np
from pydub import AudioSegment
from pydub.utils import which
from pydub import AudioSegment
import os

# Use full paths with double backslashes or raw strings
ffmpeg_path = r"C:\Users\visha\Downloads\ffmpeg\bin\ffmpeg.exe"
ffprobe_path = r"C:\Users\visha\Downloads\ffmpeg\bin\ffprobe.exe"

# Ensure files exist
if not os.path.isfile(ffmpeg_path):
    raise FileNotFoundError(f"ffmpeg not found at {ffmpeg_path}")
if not os.path.isfile(ffprobe_path):
    raise FileNotFoundError(f"ffprobe not found at {ffprobe_path}")

# Assign to pydub
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path





class EmpathyEngine:
    def __init__(self):
        try:
            self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
            self.use_hf = True
        except Exception:
            from nltk.sentiment import SentimentIntensityAnalyzer
            nltk.download("vader_lexicon", quiet=True)
            self.classifier = SentimentIntensityAnalyzer()
            self.use_hf = False

    def detect_emotion(self, text):
        if self.use_hf:
            results = self.classifier(text)
            results = sorted(results[0], key=lambda x: x['score'], reverse=True)
            return results[0]['label'], results[0]['score']
        else:
            scores = self.classifier.polarity_scores(text)
            if scores["compound"] >= 0.05:
                return "joy", scores["compound"]
            elif scores["compound"] <= -0.05:
                return "anger", abs(scores["compound"])
            else:
                return "neutral", 0.5

    def emotion_to_params(self, emotion, intensity):
        mapping = {
            "joy": {"pitch": 3, "rate": 1.2, "volume": 2},
            "anger": {"pitch": -3, "rate": 1.3, "volume": 4},
            "sadness": {"pitch": -4, "rate": 0.8, "volume": -3},
            "neutral": {"pitch": 0, "rate": 1.0, "volume": 0},
        }
        params = mapping.get(emotion, mapping["neutral"])
        return {
            "pitch": int(params["pitch"] * intensity),
            "rate": params["rate"] * (0.8 + 0.4 * intensity),
            "volume": int(params["volume"] * intensity),
        }

    def apply_modulation(self, audio_path, params):
        sound = AudioSegment.from_file(audio_path, format="mp3")
        sound = sound + params["volume"]
        new_frame_rate = int(sound.frame_rate * params["rate"])
        sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_frame_rate})
        sound = sound.set_frame_rate(44100)
        n_steps = params["pitch"]
        new_sample_rate = int(sound.frame_rate * (2.0 ** (n_steps / 12.0)))
        sound = sound._spawn(sound.raw_data, overrides={"frame_rate": new_sample_rate})
        return sound.set_frame_rate(44100)

    def process_text(self, text, output_dir="./outputs"):
        emotion, intensity = self.detect_emotion(text)
        params = self.emotion_to_params(emotion, intensity)
        tts = gTTS(text=text, lang="en")
        temp_path = os.path.join(output_dir, "temp.mp3")
        tts.save(temp_path)
        modulated = self.apply_modulation(temp_path, params)
        output_path = os.path.join(output_dir, f"output_{emotion}.wav")
        modulated.export(output_path, format="wav")
        return output_path
