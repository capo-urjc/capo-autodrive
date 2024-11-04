import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

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
