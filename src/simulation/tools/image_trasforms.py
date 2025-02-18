import numpy as np

DATASET_MEAN = np.array([0.5435199558323847, 0.5386219601332897, 0.5325046995624928], dtype=np.float32)
DATASET_STD = np.array([0.1990361211957141, 0.20639664185617518, 0.22376878168172593], dtype=np.float32)

def normalize(image: np.ndarray) -> np.ndarray:
    return image / 255.0

def standardize(image: np.ndarray) -> np.ndarray:
    return (image - DATASET_MEAN) / DATASET_STD