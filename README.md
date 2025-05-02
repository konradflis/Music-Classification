# Music-Classification

## Overview

Music Genre Classification using a Convolutional Neural Network (CNN).

---

## Project Structure

- **Dataset**: Place the dataset in the following structure:
  ```
  dataset/
  ├── fma_small/
  └── fma_metadata/
  ```
- **Metadata**: The file `project_data/tracks.csv` contains the list of all tracks in the dataset and their IDs.
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

Select created virtual enviroment in your IDE or activate it using:

```bash
poetry shell
```

---

## Usage

### 1. Extract Spectrograms

Run the spectrogram extraction script spectrogram_extract.py to generate spectrograms.

The spectrograms will be saved in the `project_data/spectrograms/` folder.

### 2. Load Individual Spectrograms

Use the utility function from `utils.py` to load individual spectrograms. Pass the path to the spectrograms folder and the track ID as arguments.

---

## Notes

- Ensure the dataset is correctly placed in the `dataset/` folder before running any scripts.
- For more details on the dataset, refer to the `fma_metadata` documentation.

---
