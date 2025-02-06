import os

import cv2
import numpy as np
import math

cameras = [
    {
        'CameraForward':
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': 900, 'image_size_y': 256, 'fov': 100,
                'x': 0.7, 'y': 0.0, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            }
    },
    {
        'CameraRightBackward':
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': 900, 'image_size_y': 256, 'fov': 100,
                'x': 1.2, 'y': 0.95, 'z': 0.75, 'roll': 0.0, 'pitch': 0.0, 'yaw': 140.0
            }
    },
    {
        'CameraLeftBackward':
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': 900, 'image_size_y': 256, 'fov': 100,
                'x': 1.2, 'y': -0.95, 'z': 0.75, 'roll': 0.0, 'pitch': 0.0, 'yaw': -140.0
            }
    },
    {
        'CameraRightForward':
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': 900, 'image_size_y': 256, 'fov': 100,
                'x': 0.0, 'y': 0.95, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0
            }
    },
    {
        'CameraLeftForward':
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': 900, 'image_size_y': 256, 'fov': 100,
                'x': 0.0, 'y': -0.95, 'z': 1.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0
            }
    },
    {
        'CameraBackward':
            {
                'bp': 'sensor.camera.rgb',
                'image_size_x': 900, 'image_size_y': 256, 'fov': 100,
                'x': -2.4, 'y': 0.0, 'z': 1.10, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0
            }
    }
]

# Assume each camera's image is captured and saved as separate variables
def capture_images(source_folder: str, frame: int):
    file_name = f"{frame}.png"
    camera_names =   [ list(k.keys())[0] for k in cameras ]

    result = { name: cv2.imread(os.path.join(source_folder, name, file_name)) for name in camera_names }
    return result


# Function to define the transformation points for each camera based on the camera setup
def get_homography_points(camera_params, camera_name):
    # Define the source points from the original camera image
    src_points = np.float32([[0, 0], [camera_params['image_size_x'], 0],
                             [camera_params['image_size_x'], camera_params['image_size_y']],
                             [0, camera_params['image_size_y']]])

    # Extract camera parameters
    x = camera_params['x']
    y = camera_params['y']
    z = camera_params['z']
    yaw = camera_params['yaw']

    # Calculate the destination points in the BEV canvas
    # Define scaling factors for adjusting the final BEV canvas
    scaling_factor = 400  # Adjust this to fit all points into the BEV canvas

    # Convert yaw to radians for calculations
    yaw_rad = math.radians(yaw)

    # Calculate the positions of the four corners in the BEV plane based on camera orientation
    dst_points = np.float32([
        [
            x + scaling_factor * math.cos(yaw_rad) - scaling_factor * math.sin(yaw_rad),
            y + scaling_factor * math.sin(yaw_rad) + scaling_factor * math.cos(yaw_rad)
        ],
        [
            x + scaling_factor * math.cos(yaw_rad) + scaling_factor * math.sin(yaw_rad),
            y + scaling_factor * math.sin(yaw_rad) - scaling_factor * math.cos(yaw_rad)
        ],
        [
            x - scaling_factor * math.cos(yaw_rad) + scaling_factor * math.sin(yaw_rad),
            y - scaling_factor * math.sin(yaw_rad) - scaling_factor * math.cos(yaw_rad)
        ],
        [
            x - scaling_factor * math.cos(yaw_rad) - scaling_factor * math.sin(yaw_rad),
            y - scaling_factor * math.sin(yaw_rad) + scaling_factor * math.cos(yaw_rad)
        ]
    ])

    return src_points, dst_points


# Function to warp image based on homography
def warp_image(image, src_points, dst_points):
    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, homography_matrix, (1200, 800))
    return warped_image


# Generate BEV
def generate_bev(camera_setup):
    # Capture all images
    images = capture_images('/raid/code/dmariaa/capo-autodrive/data/dataset/weather1/Accident_2',
                            40)

    # Create an empty canvas for the BEV (adjust dimensions as needed)
    bev_image = np.zeros((800, 1200, 3), dtype=np.uint8)

    # Iterate through cameras and compute BEV
    for camera in camera_setup:
        for camera_name, params in camera.items():
            # Extract the actual image
            image = images[camera_name]

            # Get the source and destination points based on the camera setup
            src_points, dst_points = get_homography_points(params, camera_name)

            # Warp the image to the BEV perspective
            warped_image = warp_image(image, src_points, dst_points)

            # Blend warped image into BEV canvas
            mask = warped_image > 0
            bev_image[mask] = warped_image[mask]

    # Show the final BEV image
    # cv2.imshow("Bird's Eye View", bev_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join("/raid/code/dmariaa/capo-autodrive/output", "custom_bev.jpg"), bev_image)

# Use the camera setup from the canvas
generate_bev(cameras)
