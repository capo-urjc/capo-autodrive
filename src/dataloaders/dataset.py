import os
import glob

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize
from .utils import sensor_code
import json

from .transformations import RecordsTransform, ImageNormalization
from torchvision import transforms

class AutodriveDataset(Dataset):
    """
    Clase para cargar un dataset personalizado basado en un archivo CSV de configuración.
    """

    def __init__(self, csv_file, subset='train', seq_len=5, transform=None, sensors=['rgb_f'], use_encoded_images=False):
        """
        Inicializa el dataset.

        Parameters:
        - csv_file (str): Path to the csv folder with route loading information
        - subset (str): Subset to be used: 'train' or 'test'
        - seq_len (int): Sequence length
        - transform (callable, optional): Transformation to apply to data
        - sensors (list(str)): List of sensors to load from src.dataloaders.utils import sensor_code
        """
        self.subset = 'TRAIN' if subset.lower() == 'train' else 'TEST'
        self.seq_len = seq_len
        self.transform = transform
        self.data_frame = self.process_dataframe(csv_file)
        self.n_samples =  self.data_frame['File Count'].sum() - (seq_len)

        self.sensors = sensors
        self.use_encoded_images = use_encoded_images
        if use_encoded_images:
            self.backbone_name = 'dinov2_vitb14_reg_lc.npy'

    def process_dataframe(self, csv_file):
        data_frame = pd.read_csv(csv_file)
        data_frame = data_frame[data_frame['Select'] == self.subset]
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
        start_index_within_folder = int(idx - (selected_row['Cumulative Count'] - folder_file_count) - 1)

        # Adjust the starting index if it exceeds the folder boundary when adding seq_len
        if start_index_within_folder + self.seq_len >= folder_file_count:
            start_index_within_folder = folder_file_count - self.seq_len - 1 # Shift to the last possible sequence within the folder

        # Define the sequence range
        sequence_indices = list(range(start_index_within_folder, start_index_within_folder + self.seq_len))

        # Create a dictionary with each sensor key pointing to an empty list
        keys = {sensor: None for sensor in self.sensors}

        for key in keys:

            if key not in sensor_code.keys():
                raise KeyError('A unavailable sensor has been queried. Check the list of sensors: ' + str(sensor_code.keys()))

            data_path = os.path.join(folder_path, sensor_code[key])
            if 'rgb' in key:
                seq = self.load_rgb(data_path, sequence_indices, self.use_encoded_images)
            elif 'gnss' in key or 'imu' in key:
                seq = self.load_gnss_imu(data_path, sequence_indices)
            elif 'lidar' in key:
                seq = self.load_lidar(data_path, sequence_indices)
            elif 'radar' in key:
                # TODO: Check how to load radar info since it contains several rows for each rgb frame
                seq = self.load_radar(data_path, sequence_indices)
            elif 'records' in key:
                # records = json.load(open(data_path))['records']
                # seq = [records[i] for i in sequence_indices]
                records = pd.read_csv(data_path)
                increased_sequence_indices = [x + 1 for x in sequence_indices]
                seq = records.iloc[increased_sequence_indices]

            keys[key] = seq

        if self.transform:
            keys = self.transform(keys)

        return keys

    def load_rgb(self, path, sequence_indices, use_coded_images=False):

        if use_coded_images:
            seq = np.load(path+'/'+self.backbone_name)
            seq = seq[sequence_indices]
        else:
            seq = []
            for i in sequence_indices:
                image = Image.open(os.path.join(path, str(i) + '.png'))
                image = image.convert("RGB")
                image_array = np.array(image).transpose((2, 0, 1))
                seq.append(image_array)
            seq = np.array(seq)

        return seq

    def load_gnss_imu(self, path, sequence_indices):
        data_frame = self._load_from_dataframe(path)
        selected_rows = data_frame.iloc[sequence_indices]
        return selected_rows.to_numpy()

    def load_lidar(self, path, sequence_indices):
        header_lines = 10
        columns = ["x", "y", "z", "I"]

        seq = []
        for i in sequence_indices:
            # Read the data with pandas, skipping the header lines
            df = pd.read_csv(os.path.join(path, str(i) + '.ply'), skiprows=header_lines, delim_whitespace=True, names=columns)
            seq.append(df.to_numpy())
        return seq

    def load_radar(self, path, sequence_indices):
        seq = []
        for i in sequence_indices:
            csv = pd.read_csv(os.path.join(path, str(i) + '.csv'))
            seq.append(np.median(csv.to_numpy(), axis=0))
        return np.array(seq)

    def _load_from_dataframe(self, path):
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        data_frame = pd.read_csv(csv_files[0])  # Assuming there is only one csv
        return data_frame

if __name__ == "__main__":
    csv_file = "src/dataloaders/csv/config_folders.csv"
    # dataset = AutodriveDataset(csv_file, seq_len=5, transform=None, sensors=['rgb_f', 'rgb_lf', 'rgb_rf', 'rgb_bev', 'gnss', 'imu', 'lidar', 'radar'])
    transform = transforms.Compose([
        ImageNormalization(),
        RecordsTransform(),
    ])
    # dataset = AutodriveDataset(csv_file, seq_len=10, transform=transform, sensors=['rgb_f', 'rgb_lf', 'rgb_rf', 'rgb_lb', 'rgb_rb', 'rgb_b', 'records'], use_encoded_images=False)
    dataset = AutodriveDataset(csv_file, seq_len=10, transform=transform, sensors=['rgb_f', 'records'], use_encoded_images=True)

    #
    random = np.random.randint(0, dataset.__len__())
    data1 = dataset.__getitem__(random)
    print(f'Length of dataset: {dataset.__len__()}')

    dl = DataLoader(dataset, batch_size=16, shuffle=False)

    import time
    from tqdm import tqdm
    start_time = time.time()  # Start timer

    for batch in tqdm(dl):
        img = batch['rgb_f']
        wps = batch['wps']

    end_time = time.time()  # End timer

    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.6f} seconds")



