import os
import pandas as pd


def get_audio_path(audio_dir, track_name):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 'blues.00000')
    '../data/genres_original/blues/blues.00000.wav'  # Corrected example output

    """
    genre = track_name.split('.')[0]
    id = track_name.split('.')[1]  # Extract the part before the dot
    return os.path.join(audio_dir, genre, genre + '.' + id + '.wav')


def get_spectrogram_path(spectrogram_dir, track_name):
    """
    Return the path to the spectrogram given the directory where the
    spectrogram is stored and the track ID.

    Examples
    --------
    >>> import utils
    >>> SPECTROGRAM_DIR = os.environ.get('SPECTROGRAM_DIR')
    >>> utils.get_spectrogram_path(SPECTROGRAM_DIR, 'blues.00000')  # Fixed typo
    '../data/genres_original/blues/blues.00000.png'

    """
    genre = track_name.split('.')[0]
    id = track_name.split('.')[1]  # Extract the part before the dot
    return os.path.join(spectrogram_dir, genre, genre + '.' + id + '.png')
