import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import utils
import logging

# Configure logging
logging.basicConfig(filename='mfcc_extract.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

tracks = pd.read_csv('project_data/tracks_fma.csv', index_col=0)
track_ids = tracks.index

AUDIO_DIR = 'dataset/fma_small'

n_mfcc = 13
multi_col_index = pd.MultiIndex.from_product(
    [range(n_mfcc), ['mean', 'min', 'max']], names=['mfcc_coeff', 'stat'])
mfcc_df = pd.DataFrame(index=track_ids, columns=multi_col_index)


def process_track(track_id):
    try:
        path = utils.get_audio_path(AUDIO_DIR, track_id)
        if not os.path.exists(path):
            return track_id, None  # File doesn't exist

        y, sr = librosa.load(path, sr=None)
        if len(y) == 0:
            return track_id, None  # Empty audio

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] == 0:
            return track_id, None  # Failed to extract

        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        data = np.stack([mfcc_mean, mfcc_min, mfcc_max], axis=1).flatten()
        return track_id, data
    except Exception as e:
        logging.error(f"Error processing track {track_id}: {e}")
        return track_id, None


if __name__ == '__main__':
    # Run in parallel with limited workers
    # Limit to 8 workers or available CPUs
    max_workers = min(8, os.cpu_count() or 1)
    print(f"Using {max_workers} workers for parallel processing.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(process_track, track_ids), total=len(track_ids)))

    # Fill DataFrame
    for track_id, data in results:
        if data is not None:
            mfcc_df.loc[track_id] = data

    mfcc_df.to_csv('MLP/mfcc_data_fma_v1.csv', index=True)
