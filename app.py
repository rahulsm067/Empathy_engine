from flask import Flask, request, render_template, send_file
import os
from empathy_engine import EmpathyEngine

app = Flask(__name__)
engine = EmpathyEngine()

@app.route("/", methods=["GET", "POST"])
def index():
    audio_file = None
    if request.method == "POST":
        text = request.form["text"]
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        audio_file = engine.process_text(text, output_dir)
    return render_template("index.html", audio_file=audio_file)

@app.route("/play/<filename>")
def play(filename):
    return send_file(f"./outputs/{filename}", mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True)
