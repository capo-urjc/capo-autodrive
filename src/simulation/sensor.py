from __future__ import annotations

import numpy as np
import torch

from src.simulation.tools.image_trasforms import normalize, standardize
from src.simulation.tools import sim_logger
from src.dataloaders.encode_images_dataset import Dinov2Enc
import os.path
from queue import Queue, Empty
from threading import Thread, Lock

import carla
from box import Box

class Sensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = Dinov2Enc().to(device)

    def __init__(self, destination_folder: str, kwargs: dict):
        self.logger = sim_logger.get_logger(self.__class__.__name__)

        sensor_config = kwargs.copy()
        self.name = sensor_config.pop('name')
        self.bp = sensor_config.pop('bp')
        self.x = sensor_config.pop('x')
        self.y = sensor_config.pop('y')
        self.z = sensor_config.pop('z')
        self.roll = sensor_config.pop('roll')
        self.pitch = sensor_config.pop('pitch')
        self.yaw = sensor_config.pop('yaw')
        self._dino_encode = sensor_config.pop('dino_encode', False) if self.bp == "sensor.camera.rgb" else False
        self._dino_data = []

        self.attributes = Box(sensor_config.items())

        self._actor = None
        self._callback_instance = None
        self._destination_folder = os.path.join(destination_folder, self.name)
        os.makedirs(self._destination_folder, exist_ok=True)

    def __str__(self):
        return f"{self.name}: {self.bp}"

    @property
    def tag(self):
        return self._destination_folder

    def setup(self, parent: carla.Actor, client: carla.Client, provider: SensorQueuedData):
        world: carla.World = client.get_world()

        # setup sensor actor blueprint
        bp_library = world.get_blueprint_library()
        blueprint: carla.ActorBlueprint = bp_library.find(self.bp)
        for key, value in self.attributes.items():
            blueprint.set_attribute(str(key), str(value))

        # setup sensor actor transform
        transform: carla.Transform = carla.Transform(carla.Location(x=self.x, y=self.y, z=self.z),
                                                carla.Rotation(pitch=self.pitch, roll=self.roll, yaw=self.yaw))

        # spawn actor
        self._actor: carla.Actor = world.spawn_actor(blueprint, transform, parent)
        self._callback_instance = SensorCallback(self.name, self, provider)
        self._actor.listen(self._callback_instance)

    def destroy(self):
        if self._dino_encode:
            d = np.array(self._dino_data)
            file = f"{self._destination_folder}/dinov2_vitb14_reg_lc.npy"
            np.save(file, d)

        if self._actor.is_listening:
            self._actor.stop()

        if self._callback_instance:
            del self._callback_instance

        self._actor.destroy()
        self.logger.debug(f"Sensor {self.name} released")

    def save_data(self, data: object, frame: int):
        if isinstance(data, carla.Image):
            file = f"{self._destination_folder}/{frame}.png"
            if self.bp == 'sensor.camera.semantic_segmentation':
                data.save_to_disk(file, carla.ColorConverter.CityScapesPalette)
            else:
                if self._dino_encode:
                    self._dino_encode_image(data, frame)
                else:
                    data.save_to_disk(file)
        elif isinstance(data, carla.LidarMeasurement):
            sensor_file = f"{self._destination_folder}/{frame}.ply"
            data.save_to_disk(sensor_file)
        elif isinstance(data, carla.SemanticLidarMeasurement):
            sensor_file = f"{self._destination_folder}/{frame}.ply"
            data.save_to_disk(sensor_file)
        elif isinstance(data, carla.RadarMeasurement):
            sensor_file = f"{self._destination_folder}/{frame}.csv"
            data_txt = f"Altitude,Azimuth,Depth,Velocity\n"
            for point_data in data:
                data_txt += f"{point_data.altitude},{point_data.azimuth},{point_data.depth},{point_data.velocity}\n"
            with open(sensor_file, 'w') as data_file:
                data_file.write(data_txt)
        elif isinstance(data, carla.GnssMeasurement):
            sensor_file = f"{self._destination_folder}/gnss_data.csv"
            if not os.path.exists(sensor_file):
                with open(sensor_file, 'w') as data_file:
                    header_txt = f"Frame,Altitude,Latitude,Longitude\n"
                    data_file.write(header_txt)
            with open(sensor_file, 'a') as data_file:
                data_txt = f"{frame},{data.altitude},{data.latitude},{data.longitude}\n"
                data_file.write(data_txt)
        elif isinstance(data, carla.IMUMeasurement):
            sensor_file = f"{self._destination_folder}/imu_data.csv"
            if not os.path.exists(sensor_file):
                with open(sensor_file, 'w') as data_file:
                    header_txt = (f"Frame,Accelerometer X,Accelerometer y,Accelerometer Z,Compass,"
                                f"Gyroscope X,Gyroscope Y,Gyroscope Z\n")
                    data_file.write(header_txt)
            with open(sensor_file, 'a') as data_file:
                data_txt = (f"{frame},{data.accelerometer.x},{data.accelerometer.y},{data.accelerometer.z},"
                            f"{data.compass},"
                            f"{data.gyroscope.x},{data.gyroscope.y},{data.gyroscope.z}\n")
                data_file.write(data_txt)
        else:
            raise RuntimeError(f"Sensor {self.name} data type {type(data)} can not be handled.")

    def _dino_encode_image(self, image_data, frame):
        # reshape, HWC->CHW, to tensor
        data_tensor = torch.from_numpy(self._transform_image(image_data))

        # BGR->RGB, add batch and sequence dimensions
        data_tensor = data_tensor.permute(2, 0, 1)[None, None, ...]
        data_tensor = data_tensor.to(Sensor.device)
        Sensor.encoder_model.eval()
        embedding = Sensor.encoder_model(data_tensor)
        self._dino_data.append(embedding.detach().cpu().numpy())

    def _transform_image(self, image_data):
        data = np.array(image_data.raw_data, dtype=np.float32)
        data = data.reshape(image_data.height, image_data.width, -1)
        data = data[:, :, :3]
        data = data[:, :, ::-1]
        data = standardize(data)
        return data

class SensorCallback(object):
    def __init__(self, tag: str, sensor: Sensor, provider: SensorQueuedData):
        self.logger = sim_logger.get_logger(self.__class__.__name__)

        self._provider = provider
        self._tag = sensor.name
        self._sensor = sensor
        self._provider.register_sensor(self._tag, sensor)

    def __call__(self, data):
        self.logger.verbose(f"Received data for sensor {self._tag}, frame {data.frame}, "
                            f"type: {data.__class__.__name__}")
        self._provider.update_sensor(self._tag, data, data.frame)

    def __del__(self):
        self._provider.unregister_sensor(self._tag)


class SensorReceivedNoData(Exception):
    pass


class SensorQueuedData(object):
    def __init__(self, max_save_threads: int, buffer_timeout: int = 5):
        self.logger = sim_logger.get_logger(self.__class__.__name__)

        self._buffer = Queue()
        self._buffer_timeout = buffer_timeout
        self._sensor_objects = {}
        self._max_save_threads = max_save_threads

    def register_sensor(self, tag, sensor: Sensor):
        self._sensor_objects[tag] = sensor
        self.logger.debug(f"Sensor {sensor.name} registered as {tag}")

    def unregister_sensor(self, tag):
        sensor = self._sensor_objects[tag]
        del self._sensor_objects[tag]
        self.logger.debug(f"Sensor {sensor.name} [{tag}] unregistered")

    def clear_sensors(self):
        self._sensor_objects.clear()

    def update_sensor(self, tag, data, frame):
        self._buffer.put((tag, data, frame))

    def save_sensors(self, frame: int, initial_frame: int):
        current_threads: int = 0
        lock: Lock = Lock()

        def _save_sensor(sensor, frame, data):
            nonlocal current_threads
            with lock:
                current_threads += 1
            sensor.save_data(data, frame)
            with lock:
                current_threads -= 1

        pending_sensors = len(self._sensor_objects.keys())
        threads = []

        while pending_sensors > 0:
            try:
                (tag, data, data_frame) = self._buffer.get(True, self._buffer_timeout)
                if data_frame != frame:
                    continue
                pending_sensors -= 1
            except Empty:
                raise SensorReceivedNoData("A sensor took too long to send its data")

            sensor = self._sensor_objects[tag]
            # thread = Thread(target=_save_sensor, args=(sensor, data_frame - initial_frame, data))
            # threads.append(thread)
            # thread.start()
            #
            # if current_threads > self._max_save_threads:
            #     for t in threads:
            #         t.join()
            #     threads.clear()
            sensor.save_data(data, frame)

        # Finish pending threads
        for t in threads:
            t.join()
        threads.clear()

