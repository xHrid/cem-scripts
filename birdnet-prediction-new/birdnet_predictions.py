import os
import re
import sys
import argparse
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from collections import defaultdict
from scipy.special import softmax
from scipy.stats import entropy
from scipy.signal import spectrogram
import gc
np.complex = complex  # Monkey patch for compatibility
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from io import StringIO
import tempfile
from tqdm import tqdm

TARGET_SR = 48000
SEGMENT_DURATION = 120.0  # 120 seconds
SKIP_DURATION = 60    # skip 1 minute after each segment
TOTAL_SEGMENTS = 2      # desired number of 2-min samples
SNR_DB = 18

DATASETS = [r"E:\raw_data\audio_raw\spot_1_original_spot\250517-250529\30R30W", r"E:\raw_data\audio_raw\spot_1_original_spot\250602-250613\30R30W", r"E:\raw_data\audio_raw\spot_1_original_spot\250621-250705\5R5W", r"E:\raw_data\audio_raw\spot_1_original_spot\250708-250711\2R4W", r"E:\raw_data\audio_raw\spot_1_original_spot\250714-250720\2R4W", r"E:\raw_data\audio_raw\spot_1_original_spot\250720-250729\2R4W", r"E:\raw_data\audio_raw\spot_1_original_spot\250810-250825\2R4W", r"E:\raw_data\audio_raw\spot_1_original_spot\250831-250905\2R4W", r"E:\raw_data\audio_raw\spot_1_original_spot\250920-250925\2R4W", r"E:\raw_data\audio_raw\spot_1_original_spot\251012-251017\2R4W"]



def segment_audio(audio, folder_type, fs=48000):
    """
    Extracts 2-minute segments based on folder-specific recording schedules.
    """
    segments = []
    total_samples = len(audio)
    two_min_samples = int(120 * fs) # 2 minutes

    # --- Case 1: 2R4W (2 min Record, 4 min Wait) ---
    if "2R4W" in folder_type:
        if total_samples >= two_min_samples:
            segments.append(audio[:two_min_samples])
            
    # --- Case 2: 5R5W (5 min Record, 5 min Wait) ---
    elif "5R5W" in folder_type:
        if "first_last" in folder_type:
            segments.append(audio[:two_min_samples])
            if total_samples >= two_min_samples:
                segments.append(audio[-two_min_samples:])
        elif "central" in folder_type:
            start = (total_samples // 2) - (two_min_samples // 2)
            end = start + two_min_samples
            if start >= 0 and end <= total_samples:
                segments.append(audio[start:end])

    # --- Case 3: 30R30W (30 min Record, 30 min Wait) ---
    elif "30R30W" in folder_type:
        num_chunks = 10
        if total_samples >= (num_chunks * two_min_samples):
            gap = (total_samples - (num_chunks * two_min_samples)) // (num_chunks - 1)
            for i in range(num_chunks):
                start = i * (two_min_samples + gap)
                end = start + two_min_samples
                if end <= total_samples:
                    segments.append(audio[start:end])
        else:
            for start in range(0, total_samples, two_min_samples):
                end = start + two_min_samples
                if end <= total_samples:
                    segments.append(audio[start:end])
                    
    return segments if segments else None


def remove_static_noise(audio, noise_ref, sr=48000, snr_db=18):
    """Remove static noise using time-domain subtraction and spectral gating."""
    if len(noise_ref) > len(audio):
        noise_ref = noise_ref[:len(audio)]
    else:
        noise_ref = np.pad(noise_ref, (0, len(audio) - len(noise_ref)), 'wrap')
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise_ref ** 2)
    desired_noise_power = audio_power / (10 ** (snr_db / 10))
    noise_ref_scaled = noise_ref * np.sqrt(desired_noise_power / noise_power)
    audio_td = audio - noise_ref_scaled
    stft = librosa.stft(audio_td, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_stft = librosa.stft(noise_ref, n_fft=2048, hop_length=512)
    noise_mag = np.abs(noise_stft)
    noise_threshold = np.mean(noise_mag, axis=1, keepdims=True) * 1.2
    gated_mag = np.where(magnitude > noise_threshold, magnitude, 0)
    cleaned_stft = gated_mag * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=512)


def analyze_bird_audio(audio_path, lat, lon):
    """Analyze audio with BirdNET."""
    analyzer = Analyzer()
    recording = Recording(
        analyzer,
        audio_path,
        lat=lat,
        lon=lon,
    )
    recording.analyze()
    df = pd.DataFrame.from_records(recording.detections)

    if df.empty:
        return df

    df = df.rename(columns={
        "start": "start_time",
        "end": "end_time",
        "species": "species",
        "confidence": "confidence"
    })

    return df


def extract_year_month_date_hour_and_minute(filename):
    """Extract datetime components from filename."""
    match_date = re.search(r'_(\d{8})_', filename)
    match = re.search(r'_(\d{6})\.wav$', filename)
    if match and match_date:
        time_str = match.group(1)
        date_str = match_date.group(1)
        year = date_str[:4]
        month = date_str[4:6]
        day = date_str[6:]
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        return year, month, day, hour, minute
    return None, None, None, None, None


def extract_path_info(dataset_path):
    """Extract spot, date range, and code from dataset path."""
    parts = dataset_path.replace('\\', '/').split('/')
    code = parts[-1] if len(parts) > 0 else ""
    date_range = parts[-2] if len(parts) > 1 else ""
    spot_folder = parts[-3] if len(parts) > 2 else ""
    spot_match = re.search(r'spot[_\s]*(\d+)', spot_folder, re.IGNORECASE)
    spot = f"spot{spot_match.group(1)}" if spot_match else "spot_unknown"
    
    return spot, date_range, code


def main(datasets, static_noise_path, output_dir, lat=28.53, lon=77.18, target_sr=48000, snr_db=18):
    """
    Main processing function for BirdNET predictions.
    
    Args:
        datasets (list): List of dataset directory paths
        static_noise_path (str): Path to static noise reference file
        output_dir (str): Output directory for CSV files
        lat (float): Latitude for BirdNET analysis
        lon (float): Longitude for BirdNET analysis
        target_sr (int): Target sample rate
        snr_db (int): Signal-to-noise ratio in dB
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load static noise
    noise_clip, _ = librosa.load(static_noise_path, sr=target_sr)
    
    # Process each dataset
    for i in range(len(datasets)):
        all_detections = []
        
        spot, date_range, code = extract_path_info(datasets[i])
        output_filename = f"{spot}_{date_range}_{code}_classification.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing dataset {i+1}/{len(datasets)}: {spot}_{date_range}_{code}")

        # Determine folder type
        if code == "2R4W":
            folder_type_base = "2R4W"
        elif code == "5R5W":
            folder_type_base = "5R5W"
        elif code == "30R30W":
            folder_type_base = "30R30W"
        else:
            folder_type_base = "DEFAULT"

        file_counter_5r5w = 0
        
        # Loop through all .wav files
        for fname in tqdm(sorted(os.listdir(datasets[i]))):
            if fname.lower().endswith(".wav"):
                filepath = os.path.join(datasets[i], fname)
                audio, _ = librosa.load(filepath, sr=target_sr)
                sr = target_sr

                sampling_rule = folder_type_base
                if folder_type_base == "5R5W":
                    if file_counter_5r5w == 0 or file_counter_5r5w == 2:
                        sampling_rule = "5R5W_first_last"
                    else:
                        sampling_rule = "5R5W_central"
                    file_counter_5r5w = (file_counter_5r5w + 1) % 3

                audio_denoised = remove_static_noise(audio, noise_clip, sr=sr, snr_db=snr_db)
                segments = segment_audio(audio_denoised, sampling_rule, fs=sr)

                if segments is None or len(segments) == 0:
                    print(f"No segments created for file: {fname}")
                    continue
                
                year, month, day, hour, minute = extract_year_month_date_hour_and_minute(fname)
                if None in [year, month, day]:
                    print(f"Skipping file due to unmatched filename format: {fname}")
                    continue
                
                total_detections = 0
                
                # Process each segment
                for j, segment in enumerate(segments):
                    temp_segment_path = None
                    old_stdout = sys.stdout
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            temp_segment_path = tmp.name
                        sf.write(temp_segment_path, segment, target_sr)

                        old_stdout = sys.stdout
                        sys.stdout = StringIO()
                        
                        detections_df = analyze_bird_audio(
                            audio_path=temp_segment_path,
                            lat=lat,
                            lon=lon
                        )

                        sys.stdout = old_stdout
                        
                        detections_df["filename"] = fname
                        detections_df["segment_index"] = j
                        detections_df["sampling_rule"] = sampling_rule
                        detections_df["year"] = year
                        detections_df["month"] = month
                        detections_df["day"] = day
                        detections_df["hour"] = hour
                        detections_df["minute"] = minute
                        all_detections.append(detections_df)
                        
                        total_detections += len(detections_df)
                        
                    except Exception as e:
                        sys.stdout = old_stdout
                        print(f"Error processing segment {j} of {fname}: {e}")
                    
                    finally:
                        if temp_segment_path and os.path.exists(temp_segment_path):
                            try:
                                os.unlink(temp_segment_path)
                            except:
                                pass
                
                print(f"Processed file: {fname} ({len(segments)} segments) with {total_detections} total detections.")
        
        # Combine and save all detections
        if all_detections:
            final_df = pd.concat(all_detections, ignore_index=True)
            final_df.to_csv(output_path, index=False)
            print(f"Saved detections to '{output_filename}'")
        else:
            print("No detections processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files with BirdNET for bird species detection.")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of dataset directory paths")
    parser.add_argument("--noise-path", required=True, help="Path to static noise reference file")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV files")
    parser.add_argument("--lat", type=float, default=28.53, help="Latitude for BirdNET analysis")
    parser.add_argument("--lon", type=float, default=77.18, help="Longitude for BirdNET analysis")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Target sample rate")
    parser.add_argument("--snr-db", type=int, default=18, help="Signal-to-noise ratio in dB")
    
    args = parser.parse_args()
    
    main(
        datasets=args.datasets,
        static_noise_path=args.noise_path,
        output_dir=args.output_dir,
        lat=args.lat,
        lon=args.lon,
        target_sr=args.sample_rate,
        snr_db=args.snr_db
    )