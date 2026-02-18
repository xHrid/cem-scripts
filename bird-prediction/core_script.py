import os
import re
import sys
import tempfile
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import argparse

# --- Configuration ---
TARGET_SR = 48000
SNR_DB = 18

# --- Helper Functions ---

def remove_static_noise(audio, noise_ref, sr=TARGET_SR, snr_db=SNR_DB):
    """Denoises audio by subtracting a scaled noise reference."""
    if len(noise_ref) > len(audio):
        noise_ref = noise_ref[:len(audio)]
    else:
        noise_ref = np.pad(noise_ref, (0, len(audio) - len(noise_ref)), 'wrap')
    
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_ref ** 2)
    
    if noise_power == 0:
        return audio # Avoid division by zero

    desired_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_ref_scaled = noise_ref * np.sqrt(desired_noise_power / noise_power)
    
    return audio - noise_ref_scaled

def extract_datetime_components(filename):
    """Extracts date and time from the standard filename format."""
    basename = os.path.basename(filename)
    match_date = re.search(r'_(\d{8})_', basename)
    match_time = re.search(r'_(\d{6})\.wav$', basename)
    if match_time and match_date:
        time_str = match_time.group(1)
        date_str = match_date.group(1)
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:]
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        return year, month, day, hour, minute
    return None, None, None, None, None

def analyze_bird_audio(audio_path, noise_clip, analyzer, lat, lon, min_conf):
    """
    Loads, denoises, and analyzes a single audio file with BirdNET.
    Returns a DataFrame of detections.
    """
    audio_raw, orig_sr = librosa.load(audio_path, sr=None)
    
    if orig_sr != TARGET_SR:
        audio_raw = librosa.resample(y=audio_raw, orig_sr=orig_sr, target_sr=TARGET_SR)
    
    final_sound = remove_static_noise(audio_raw, noise_clip, sr=TARGET_SR, snr_db=SNR_DB)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, final_sound, samplerate=TARGET_SR)
        tmp_audio_path = tmpfile.name
    
    try:
        recording = Recording(
            analyzer,
            tmp_audio_path,
            lat=lat,
            lon=lon,
            min_conf=min_conf,
        )
        recording.analyze()
        return pd.DataFrame(recording.detections)
    finally:
        os.remove(tmp_audio_path)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run BirdNET analysis on a list of audio files.")
    parser.add_argument('--input-files', nargs='+', required=True, help="List of .wav file paths to analyze.")
    parser.add_argument('--output-file', type=str, required=True, help="Path to save the combined CSV output.")
    parser.add_argument('--static-noise-file', type=str, required=True, help="Path to the static noise .wav file.")
    parser.add_argument('--lat', type=float, required=True, help="Latitude for analysis.")
    parser.add_argument('--lon', type=float, required=True, help="Longitude for analysis.")
    parser.add_argument('--min-confidence', type=float, default=0.5, help="Minimum confidence threshold.")
    
    args = parser.parse_args()

    print("Initializing BirdNET Analyzer...")
    analyzer = Analyzer()
    
    print(f"Loading static noise clip from: {args.static_noise_file}")
    try:
        noise_clip, _ = librosa.load('static_noise.wav', sr=TARGET_SR)
    except Exception as e:
        print(f"FATAL ERROR: Could not load noise file. {e}", file=sys.stderr)
        sys.exit(1)

    all_detections = []
    print(f"--- Processing {len(args.input_files)} file(s) ---")

    for filepath in args.input_files:
        fname = os.path.basename(filepath)
        year, month, day, hour, minute = extract_datetime_components(fname)
        if hour is None:
            print(f"Skipping file (unmatched date/time format): {fname}")
            continue

        try:
            detections_df = analyze_bird_audio(filepath, noise_clip, analyzer, args.lat, args.lon, args.min_confidence)

            if not detections_df.empty:
                detections_df["filename"] = fname
                detections_df["year"] = year
                detections_df["month"] = month
                detections_df["day"] = day
                detections_df["hour"] = hour
                detections_df["minute"] = minute
                all_detections.append(detections_df)
            
            print(f"  Processed: {fname} ({len(detections_df)} detections)")

        except Exception as e:
            print(f"  ERROR processing {fname}: {e}", file=sys.stderr)

    if all_detections:
        final_df = pd.concat(all_detections, ignore_index=True)
        final_df.to_csv(args.output_file, index=False)
        print(f"--- ✅ Saved detections to: {args.output_file} ---")
    else:
        print("--- No detections found in any files ---")

if __name__ == "__main__":
    main()