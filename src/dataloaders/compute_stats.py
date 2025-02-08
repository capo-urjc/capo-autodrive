import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_image(path):
    """Carga una imagen y calcula su media y cuadrados de los valores de píxeles normalizados."""
    try:
        img = Image.open(path).convert("RGB")
        pixels = np.array(img).reshape(-1, 3) / 255.0
        return np.mean(pixels, axis=0), np.mean(pixels ** 2, axis=0)
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return np.zeros(3), np.zeros(3)

def compute_mean_std(csv_file):
    """
    Calcula la media y desviación estándar por canal (R, G, B) de una lista de imágenes en paralelo.
    :param csv_file: Archivo CSV con las rutas de las carpetas de imágenes.
    :return: Diccionario con la media y desviación estándar por canal.
    """
    paths = []
    df = pd.read_csv(csv_file)
    if "Folder Path" in df.columns:
        folder_paths = df["Folder Path"].tolist()
        file_count = df["File Count"].tolist()

        for path, n_frames in zip(folder_paths, file_count):
            camera_folders = [f for f in os.listdir(path) if "Camera" in f and "CameraBEV" not in f and os.path.isdir(os.path.join(path, f))]
            for camera in camera_folders:
                paths.extend([f"{path}/{camera}/{frame}.png" for frame in range(n_frames)])
    else:
        print("Error: 'Folder Path' column not found in CSV file.")
        return None

    print(f'Total number of images: {len(paths)}')

    means = np.zeros(3)
    squares = np.zeros(3)
    num_images = len(paths)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, paths), total=num_images))

    for mean, square in results:
        means += mean
        squares += square

    mean = means / num_images
    std_dev = np.sqrt((squares / num_images) - (mean ** 2))

    return {"mean": mean.tolist(), "std_dev": std_dev.tolist()}

# Ejemplo de uso
mean_std = compute_mean_std("src/dataloaders/csv/config_folders.csv")
with open('stats.txt', 'w') as f:
    print(mean_std, file=f)
print(mean_std)
