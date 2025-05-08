import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import utils
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='spectrogram_extract.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

tracks = pd.read_csv('project_data/tracks.csv', index_col=0)
track_ids = tracks.index

AUDIO_DIR = 'dataset/fma_small'
SPECT_PATH = 'project_data/spectrograms'


def process_track(track_id):
    try:
        path = utils.get_audio_path(AUDIO_DIR, track_id)
        output_file = utils.get_spectrogram_path(SPECT_PATH, track_id)

        if not os.path.exists(path):
            return track_id, False  # File doesn't exist

        y, sr = librosa.load(path, sr=None)
        if len(y) == 0:
            return track_id, False  # Empty audio

        # Calculate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        normalized = (mel_spec_db - np.min(mel_spec_db)) / \
            (np.max(mel_spec_db) - np.min(mel_spec_db))

        # Save spectrogram as an image
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            normalized, sr=sr, hop_length=512, cmap='viridis')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        return track_id, True
    except Exception as e:
        logging.error(f"Error processing track {track_id}: {e}")
        return track_id, False


if __name__ == '__main__':
    # Run in parallel with limited workers
    max_workers = min(6, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(process_track, track_ids), total=len(track_ids)))

    # Log results
    success_count = sum(1 for _, success in results if success)
    print(f"Successfully processed {success_count}/{len(track_ids)} tracks.")
