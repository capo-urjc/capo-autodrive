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
    assert len(set(lengths)), "Camera image sequences are not the same length."

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