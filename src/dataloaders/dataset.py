import os
import glob

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.dataloaders.utils import sensor_code


class CustomDataset(Dataset):
    """
    Clase para cargar un dataset personalizado basado en un archivo CSV de configuración.
    """

    def __init__(self, csv_file, root_dir, seq_len=5, transform=None):
        """
        Inicializa el dataset.

        Parameters:
        - csv_file (str): Ruta al archivo CSV con las rutas y etiquetas.
        - root_dir (str): Ruta al directorio raíz donde se encuentran las imágenes.
        - seq_len (int): Longitud de la secuencia a cargar.
        - transform (callable, optional): Transformaciones a aplicar a las imágenes.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.frame_paths = self._get_frame_paths()

    def _get_frame_paths(self):
        """
        Crea una lista de todos los paths de frames en las carpetas seleccionadas.

        Returns:
        - list: Lista de todos los paths de frames disponibles en el dataset.
        """
        frame_paths = []
        for index, row in self.data_frame.iterrows():
            folder_path = row['Folder Path']  # Asumiendo que la columna 'Folder Path' contiene las rutas
            if row['Select'] == 'TRUE':  # Solo incluir carpetas seleccionadas
                camera_forward_path = os.path.join(folder_path, "CameraForward")
                if os.path.exists(camera_forward_path):
                    # Obtener todos los archivos de imágenes en la carpeta 'CameraForward'
                    for file_name in sorted(os.listdir(camera_forward_path)):
                        if file_name.endswith(('.jpg', '.png', '.jpeg')):  # Filtrar por formato de imagen
                            frame_paths.append(os.path.join(camera_forward_path, file_name))
        return frame_paths

    def __len__(self):
        """Devuelve el número total de secuencias en el dataset."""
        return len(self.frame_paths) - self.seq_len + 1  # Ajustar para que no se salga del rango

    def __getitem__(self, idx):
        """
        Devuelve una secuencia de frames del dataset.

        Parameters:
        - idx (int): Índice de la secuencia que se desea obtener.

        Returns:
        - dict: Contiene la secuencia de imágenes y el path de la carpeta.
        """
        # Obtener la secuencia de frames
        frames = []
        for i in range(self.seq_len):
            frame_path = self.frame_paths[idx + i]  # Cargar frames consecutivos
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # Obtener el path del primer frame en la secuencia
        first_frame_path = os.path.dirname(self.frame_paths[idx])

        return {
            'images': torch.stack(frames),  # Apilar las imágenes en un tensor
            'folder_path': first_frame_path
        }


class AutodriveDataset(Dataset):
    """
    Clase para cargar un dataset personalizado basado en un archivo CSV de configuración.
    """

    def __init__(self, csv_file, seq_len=5, transform=None, sensors=['rgb_f']):
        """
        Inicializa el dataset.

        Parameters:
        - csv_file (str): Path to the csv folder with route loading information
        - seq_len (int): Sequence length
        - transform (callable, optional): Transformation to apply to data
        - sensors (list(str)): List of sensors to load from  from src.dataloaders.utils import sensor_code
        """
        self.seq_len = seq_len
        self.transform = transform
        self.data_frame = self.process_dataframe(csv_file)
        self.n_samples =  self.data_frame['File Count'].sum() - seq_len

        self.sensors = sensors

    def process_dataframe(self, csv_file):
        data_frame = pd.read_csv(csv_file)
        data_frame = data_frame[data_frame['Select'] == True]
        data_frame['Cumulative Count'] = data_frame['File Count'].cumsum()
        return data_frame


    def __len__(self):
        """Devuelve el número total de secuencias en el dataset."""
        return self.n_samples  # Ajustar para que no se salga del rango

    def __getitem__(self, idx):
        """
        Devuelve una secuencia de frames del dataset.

        Parameters:
        - idx (int): Índice de la secuencia que se desea obtener.

        Returns:
        - dict: Contiene la secuencia de imágenes y el path de la carpeta.
        """
        idx = idx % self.n_samples

        selected_row = self.data_frame[self.data_frame['Cumulative Count'] >= idx].iloc[0]  # Find the folder where this random number falls within the cumulative count
        folder_path = selected_row['Folder Path']  # Extract the folder path and the image index within that folder
        folder_file_count = selected_row['File Count']  # Calculate the starting index within the folder
        start_index_within_folder = int(idx - (selected_row['Cumulative Count'] - folder_file_count))

        # Adjust the starting index if it exceeds the folder boundary when adding seq_len
        if start_index_within_folder + self.seq_len > folder_file_count:
            start_index_within_folder = folder_file_count - self.seq_len  # Shift to the last possible sequence within the folder

        # Define the sequence range
        sequence_indices = list(range(start_index_within_folder, start_index_within_folder + self.seq_len))

        # Create a dictionary with each sensor key pointing to an empty list
        sensor_dict = {sensor: None for sensor in self.sensors}
        for key in sensor_dict:

            if key not in sensor_code.keys():
                raise KeyError('A unavailable sensor has been queried. Check the list of sensors: ' + str(sensor_code.keys()))

            data_path = os.path.join(folder_path, sensor_code[key])
            if 'rgb' in key:
                seq = self.load_rgb(data_path, sequence_indices)
            elif 'gnss' in key or 'imu' in key:
                seq = self.load_gnss_imu(data_path, sequence_indices)
            elif 'lidar' in key:
                seq = self.load_lidar(data_path, sequence_indices)
            elif 'radar' in key:
                # TODO: Check how to load radar info since it contains several rows for each rgb frame
                seq = self.load_radar(data_path, sequence_indices)

            sensor_dict[key] = seq

        return sensor_dict

    def load_rgb(self, path, sequence_indices):
        seq = []
        for i in sequence_indices:
            image = Image.open(os.path.join(path, str(i) + '.png'))
            image_array = np.array(image)
            seq.append(image_array)

        return np.array(seq)

    def load_gnss_imu(self, path, sequence_indices):
        data_frame = self._load_from_dataframe(path)
        selected_rows = data_frame.iloc[sequence_indices]
        return selected_rows

    def load_lidar(self, path, sequence_indices):
        raise NotImplementedError("This method is not implemented yet.")

    def load_radar(self, path, sequence_indices):
        seq = []
        for i in sequence_indices:
            csv = pd.read_csv(os.path.join(path, str(i) + '.csv'))
            seq.append(csv)
        return seq


    def _load_from_dataframe(self, path):
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        data_frame = pd.read_csv(csv_files[0])  # Assuming there is only one csv
        return data_frame

if __name__ == "__main__":
    csv_file = "src/dataloaders/csv/config_folders.csv"
    dataset = AutodriveDataset(csv_file, seq_len=5, transform=None, sensors=['rgb_f', 'rgb_lf', 'rgb_rf', 'rgb_bev', 'gnss', 'imu', 'radar', 'aa'])

    random = np.random.randint(0, dataset.__len__())
    data1 = dataset.__getitem__(random)

    random = np.random.randint(0, dataset.__len__())
    data1 = dataset.__getitem__(random)

    random = np.random.randint(0, dataset.__len__())
    data1 = dataset.__getitem__(random)

    print()




