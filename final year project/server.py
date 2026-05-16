from flask import Flask, request, send_file, send_from_directory, jsonify, after_this_request
from flask_cors import CORS
import tempfile
import soundfile as sf
import noisereduce as nr
import numpy as np
import os
import subprocess
import shutil

app = Flask(__name__, static_folder="final year project", static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})


def convert_to_wav(in_path):
    """Convert any audio file to WAV using ffmpeg. Returns path to wav file."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert non-WAV files. Install ffmpeg and try again.")
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ar", "44100", "-ac", "1", out_file]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode('utf8', errors='ignore')}")
    return out_file


def read_audio_with_conversion(path, tmp_files):
    """Try to read audio with soundfile; if fails, attempt conversion with ffmpeg."""
    try:
        data, rate = sf.read(path)
        return data, rate
    except Exception:
        wav_path = convert_to_wav(path)
        tmp_files.append(wav_path)
        data, rate = sf.read(wav_path)
        return data, rate


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/project.js")
def project_js():
    return send_from_directory(".", "project.js")


@app.route("/denoise", methods=["POST"])
def denoise():
    tmp_files = []
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded = request.files["file"]
        if not uploaded or uploaded.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # save upload to a temporary file (preserve extension if present)
        suffix = os.path.splitext(uploaded.filename)[1] or ""
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_in.close()
        uploaded.save(tmp_in.name)
        tmp_files.append(tmp_in.name)

        # read audio (convert if needed)
        data, rate = read_audio_with_conversion(tmp_in.name, tmp_files)

        if data is None or len(data) == 0:
            # cleanup on error
            for p in tmp_files:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            return jsonify({"error": "Empty or unreadable audio file"}), 400

        # convert multi-channel to mono
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)

        data = data.astype(np.float32)
        rate = int(rate)

        # take a short noise sample (first 0.5s or available)
        noise_len = min(len(data), int(rate * 0.5))
        noise_sample = data[:noise_len]

        reduced = nr.reduce_noise(y=data, y_noise=noise_sample, sr=rate)

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        tmp_files.append(out_file)
        sf.write(out_file, reduced, rate)

        # ensure temporary files are removed after response is sent
        @after_this_request
        def cleanup(response):
            for p in tmp_files:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            return response

        return send_file(out_file, mimetype="audio/wav")
    except Exception as e:
        app.logger.exception("Error in /denoise")
        # cleanup on exception
        for p in tmp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
