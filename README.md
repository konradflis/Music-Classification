# Music-Classification

## Overview

Music Genre Classification using a Convolutional Neural Network (CNN).

---

## Project Structure

- **Dataset**: Place the dataset in the following structure:
  ```
  dataset/
  ├── data/
  │   └── genres_original/
  ```
- **Metadata**: The file `project_data/tracks.csv` contains the list of all tracks in the dataset and their names.
- **Spectrograms**: Spectrograms will be generated and stored in the `project_data/spectrograms/` folder.

---

## Setup Instructions

### 1. Install Poetry

Poetry is used for dependency management. Install it using the following command:

```bash
pip install poetry
```

### 2. Install Dependencies

Run the following command to install all dependencies:

```bash
poetry install
```

### 3. Activate the Virtual Environment

Select the created virtual environment in your IDE or activate it using:

```bash
poetry shell
```

### 4. Add Dependencies

To add a new dependency to the project, use the following command:

```bash
poetry add <package-name>
```

For example, to add `numpy`:

```bash
poetry add numpy
```

---

## Usage

### 1. Extract Spectrograms

Spectrogram extraction is done using the `audio_to_spectrogram_v2.ipynb` notebook. The notebook contains the code to extract spectrograms from the audio files in the dataset.  
The spectrograms will be saved in the `project_data/spectrograms/` folder.

### 2. Load Individual Spectrograms

Use the utility function from `utils.py` to load individual spectrograms. Pass the path to the spectrograms folder and the track name as arguments.

---

## Notes

- Ensure the dataset is correctly placed in the `dataset/` folder before running any scripts.
- For more details on the dataset, refer to the `fma_metadata` documentation.

---
