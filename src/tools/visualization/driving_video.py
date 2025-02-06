import json

import cv2
import os
import numpy as np
from tqdm import tqdm

cameras = [
    'CameraLeftForward',
    'CameraForward',
    'CameraRightForward',
    'CameraLeftBackward',
    'CameraBackward',
    'CameraRightBackward',
    'CameraBEV'
]


def read_ego_logs(data_folder: str):
    ego_logs_file = os.path.join(data_folder, "ego_logs.json")
    with open(ego_logs_file, 'r', encoding='utf-8') as file:
        ego_logs_data = json.load(file)
    return ego_logs_data

def to_words(snake_str):
    return " ".join(x.capitalize() for x in snake_str.lower().split("_"))


def plot_ego_data(frame: np.ndarray, ego_data: dict, top_y: int = 512):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)  # White text
    font_thickness = 1

    y_pos = top_y + 20
    x_pos = 20
    for key in ego_data['control'].keys():
        value = ego_data['control'][key]
        cv2.putText(frame, f"{to_words(key)}: {value:.2f}", (x_pos, y_pos), font, font_scale, font_color, font_thickness,
                    cv2.LINE_AA)
        y_pos += 20

        
def generate_video(data_folder: str, output_video: str):
    cameras_data = {}
    lengths = []

    for camera in cameras:
        camera_images = sorted([img for img in os.listdir(os.path.join(data_folder, camera)) if img.endswith('.png')],
                               key=lambda x: int(os.path.splitext(x)[0]))
        cameras_data[camera] = camera_images
        lengths.append(len(camera_images))

    fps = 20

    # Ensure both cameras have the same number of frames for syncing
    assert len(set(lengths))==1, "Camera image sequences are not the same length."

    ego_logs_data = read_ego_logs(data_folder=data_folder)
    assert 'records' in ego_logs_data, "Ego log data not found inside file"
    assert len(ego_logs_data['records']) == lengths[0], "Ego log data doesn't contain the correct number of frames"

    # Read the first image to get dimensions
    # 900x256 - cameras
    # 512x512 - BEV
    height, width = (256, 900)
    bev_height, bev_width = (512, 512)
    frame_height, frame_width = (height * 2 + bev_height, width * 3)
    padding_left = (frame_width - bev_width) // 2
    padding_right = frame_width - bev_width - padding_left

    # Stack frames side-by-side (you can also stack vertically by swapping axis in np.hstack)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    with tqdm(total=lengths[0]) as pbar:
        for i in range(lengths[0]):
            frame_LF =  cv2.imread(os.path.join(data_folder, 'CameraLeftForward', cameras_data['CameraLeftForward'][i]))
            frame_F =   cv2.imread(os.path.join(data_folder, 'CameraForward', cameras_data['CameraForward'][i]))
            frame_RF =  cv2.imread(os.path.join(data_folder, 'CameraRightForward', cameras_data['CameraRightForward'][i]))
            frame_LB =  cv2.imread(os.path.join(data_folder, 'CameraLeftBackward', cameras_data['CameraLeftBackward'][i]))
            frame_B =   cv2.imread(os.path.join(data_folder, 'CameraBackward', cameras_data['CameraBackward'][i]))
            frame_RB =  cv2.imread(os.path.join(data_folder, 'CameraRightBackward', cameras_data['CameraRightBackward'][i]))
            frame_BEV = cv2.imread(os.path.join(data_folder, 'CameraBEV', cameras_data['CameraBEV'][i]))

            row1 = np.hstack((frame_LF, frame_F, frame_RF))
            row2 = np.hstack((frame_LB, frame_B, frame_RB))

            # Pad the BEV frame to center it horizontally relative to the two rows
            bev_padded = cv2.copyMakeBorder(frame_BEV,
                top=0,
                bottom=0,
                left=padding_left,
                right=padding_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            final_frame = np.vstack((row1, row2, bev_padded))

            # Extract specific ego data fields for overlay
            ego_data = ego_logs_data['records'][i]
            plot_ego_data(frame=final_frame, ego_data=ego_data, top_y=512)

            if i==0:
                output_folder = os.path.dirname(output_video)
                cv2.imwrite(os.path.join(output_folder, "poster.png"), final_frame)

            video.write(final_frame)
            pbar.update(1)

    # Release the video writer
    video.release()
    print("Multi-camera video created successfully.")


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("-d", "--data-folder", type=click.types.Path(file_okay=False, dir_okay=True),
                  required=True, help="Data folder")
    @click.option("-o", "--output-file", type=click.types.Path(file_okay=True, dir_okay=False),
                  required=True, help="Output video file")
    def generate(data_folder: str, output_file: str):
        generate_video(data_folder=data_folder, output_video=output_file)

    generate()