import os
import csv
import argparse

def count_files_in_camera_forward(folder_path):
    """
    Cuenta el número de archivos en la carpeta 'CameraForward' dentro de la carpeta especificada.

    Parameters:
    - folder_path (str): Ruta de la carpeta donde se buscará 'CameraForward'.

    Returns:
    - int: Número de archivos en 'CameraForward', o 0 si no existe.
    """
    camera_forward_path = os.path.join(folder_path, "CameraForward")
    if os.path.exists(camera_forward_path) and os.path.isdir(camera_forward_path):
        # Contar archivos en la carpeta 'CameraForward'
        return len([f for f in os.listdir(camera_forward_path) if os.path.isfile(os.path.join(camera_forward_path, f))])
    return 0  # Retorna 0 si la carpeta 'CameraForward' no existe

def generate_config_csv(root_folder, output_csv="config_folders.csv"):
    """
    Genera un archivo CSV con solo las subcarpetas de segundo nivel de la carpeta raíz dada,
    ordenadas alfabéticamente, e incluye una columna de selección y el número de archivos en 'CameraForward'.

    Parameters:
    - root_folder (str): Ruta de la carpeta raíz a explorar.
    - output_csv (str): Nombre del archivo CSV de salida.
    """
    subfolders = []

    # Recorre las subcarpetas y selecciona solo las de segundo nivel
    for root, dirs, _ in os.walk(root_folder):
        # Calcula la profundidad actual en comparación con la carpeta raíz
        depth = root.count(os.sep) - root_folder.count(os.sep)

        # Solo guarda subcarpetas si están exactamente a 2 niveles de profundidad
        if depth == 1:  # Primer nivel de subcarpetas
            for dir_name in sorted(dirs):  # Ordena las carpetas dentro del primer nivel
                subfolder_path = os.path.join(root, dir_name)
                file_count = count_files_in_camera_forward(subfolder_path)  # Contar archivos en CameraForward
                subfolders.append((file_count, subfolder_path))  # Guardar el conteo y la ruta

        # Detiene el descenso en subdirectorios después del segundo nivel
        dirs[:] = [] if depth >= 1 else dirs

    # Ordena la lista de subcarpetas alfabéticamente
    subfolders.sort(key=lambda x: x[1])  # Ordenar por la ruta de las carpetas

    # Genera el archivo CSV con una columna de selección y el número de archivos
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Select", "Folder Path", "File Count"])  # Encabezado
        for file_count, subfolder in subfolders:
            writer.writerow(["TRUE", subfolder, file_count])  # Inicialmente en TRUE

if __name__ == "__main__":
    # Configuración de argumentos de línea de comando
    parser = argparse.ArgumentParser(description="Genera un archivo CSV con subcarpetas de una carpeta raíz.")
    parser.add_argument("--root_folder", type=str, default='data/dataset', help="Ruta de la carpeta raíz a explorar")
    parser.add_argument("--output_csv", type=str, default="src/data/config_folders.csv", help="Nombre del archivo CSV de salida (por defecto: config_folders.csv)")

    # Parsear los argumentos de la línea de comando
    args = parser.parse_args()

    # Llamar a la función con los argumentos
    generate_config_csv(args.root_folder, args.output_csv)
