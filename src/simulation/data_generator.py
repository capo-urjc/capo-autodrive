import logging

VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")


# Add a method to use the VERBOSE level
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


logging.Logger.verbose = verbose

import os
import pathlib
import threading
import time

from queue import Queue, Empty

import carla
import yaml
from box import Box
from carla import BlueprintLibrary, ActorBlueprint, Actor, Client, World, WorldSettings

from src.simulation.runner import CARLAServerRunner
from src.simulation.sensor import Sensor
from src.simulation.tools.replay_parser import ReplayParser


# noinspection PyArgumentList
class CarlaDataGenerator:
    def __init__(self, config_file: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config_file = config_file
        self.config = self._read_config()
        self.sensors = self._read_sensors()
        self._carla_runner = CARLAServerRunner(port=self.config.server.port)

    def _read_config(self):
        try:
            with open(self._config_file, 'r') as file:
                config = yaml.safe_load(file)
                return Box(config)  # Use Box to enable dot notation access
        except (yaml.YAMLError, FileNotFoundError, IOError) as exc:
            raise RuntimeError(f"Error reading the YAML configuration file: {exc}")

    def _read_sensors(self):
        sensors = []
        for sensor in self.config.sensors:
            sensors.append(Sensor(self.config.files.destination_folder, sensor))
        return sensors

    def _generate_recorder_info(self):
        origin_folder = self.config.files.recordings_folder
        destination_folder = self.config.files.destination_folder

        folders = [d for d in os.listdir(origin_folder) if os.path.isdir(os.path.join(origin_folder, d))]
        recorder_info = []

        for folder in folders:
            full_folder = os.path.join(origin_folder, folder)
            log_file = next(
                (file.name for file in pathlib.Path(full_folder).iterdir() if file.is_file() and file.suffix == ".log"),
                None)
            destination = os.path.splitext(log_file)[0]

            if log_file and not os.path.exists(os.path.join(destination_folder, destination)):
                recorder_info.append({
                    'folder': full_folder,
                    'name': destination,
                    'start_time': 0,
                    'duration': 0
                })

        return recorder_info

    def _connect_to_server(self, timeout: int = 20, max_retries: int = 5, retry_interval: int = 5):
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Attempt {attempt} to connect to CARLA server...")
                client = carla.Client(self.config.server.host, self.config.server.port)
                client.set_timeout(timeout)
                world = client.get_world()
                self.logger.info("Successfully connected to CARLA server.")
                return client, world
            except Exception as e:
                self.logger.debug(f"Connection attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    self.logger.debug(f"Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                else:
                    raise RuntimeError("Exceeded maximum retries. CARLA server is not ready.")

        return None, None

    def _get_recorder_data(self, recorder_file: str, client: carla.Client):
        client.set_timeout(10)
        recorder_str = client.show_recorder_file_info(os.path.abspath(recorder_file), False)
        recorder_data = ReplayParser(recorder_str)
        return recorder_data

    def _create_destination_folder(self, recorder_info: dict) -> str:
        base_destination_folder = self.config.files.destination_folder
        destination_folder = os.path.join(base_destination_folder, recorder_info['name'])

        if os.path.exists(destination_folder):
            self.logger.warning(f"Destination folder {destination_folder} already exists, this will override it's contents")
        else:
            self.logger.info(f"Data captured will be stored in {destination_folder}")
            os.makedirs(destination_folder, exist_ok=True)

        return destination_folder

    def _setup_world(self, recorder_data: ReplayParser, client: carla.Client) -> carla.World:
        client.set_timeout(240)
        # Get the world
        world: carla.World = client.load_world(recorder_data.map)
        world.tick()

        # Set world weather
        weather = carla.WeatherParameters(**self.config.weather)
        world.set_weather(weather)

        # Change fps and sync mode
        world_settings: carla.WorldSettings = world.get_settings()
        world_settings.fixed_delta_seconds = 1 / self.config.simulation.fps
        world_settings.synchronous_mode = True
        world.apply_settings(world_settings)
        world.tick()
        return world

    def _get_recording_start_end(self, recorder_info: Box, recorder_data: ReplayParser):
        start = recorder_info.start_time
        if recorder_info.duration == 0:
            end = recorder_data.duration
        elif start + recorder_info.duration > recorder_data.duration:
            self.logger.warning("Found a duration that exceeds recorder length, adjusting it to recorder length")
            end = recorder_data.duration
        elif start >= recorder_data.duration:
            self.logger.warning("Found a start point that exceeds recorder length, setting to 0")
            end = recorder_data.duration
        else:
            end = recorder_info.duration
        return start, end

    def _create_sensor_actor(self, current_sensor: Sensor,
                             parent: carla.Actor,
                             world: carla.World,
                             blueprint_lib: BlueprintLibrary) -> carla.Actor:
        transform: carla.Transform = carla.Transform(
            carla.Location(x=current_sensor.x, y=current_sensor.y, z=current_sensor.z),
            carla.Rotation(pitch=current_sensor.pitch, roll=current_sensor.roll, yaw=current_sensor.yaw))

        blueprint: carla.ActorBlueprint = blueprint_lib.find(current_sensor.bp)

        for key, value in list(dict(current_sensor.attributes).items()):
            blueprint.set_attribute(str(key), str(value))

        sensor_actor: carla.Actor = world.spawn_actor(blueprint, transform, parent)
        return sensor_actor

    def _sensor_listen(self, current_sensor: Sensor, data: object, sensor_queue: Queue):
        self.logger.verbose(f"Received data for sensor {current_sensor.name}, frame {data.frame}, "
                          f"type: {data.__class__.__name__}")
        sensor_queue.put((current_sensor, data.frame, data))
        return

    def _get_sensor_listener(self, sensor, sensor_queue):
        return lambda data: self._sensor_listen(sensor, data, sensor_queue)

    def _setup_sensors(self, world: carla.World, parent: carla.Actor, sensor_queue: Queue):
        active_sensors = []

        for sensor in self.sensors:
            self.logger.debug(f"setting up sensor {sensor.name}")
            sensor_actor: carla.Actor = self._create_sensor_actor(sensor, parent, world, world.get_blueprint_library())
            sensor_actor.listen(self._get_sensor_listener(sensor, sensor_queue))
            active_sensors.append(sensor_actor)

        return active_sensors

    def _save_sensors(self, sensor_actors: list,
                      world: World,
                      sensor_queue: Queue,
                      start_frame: int):
        current_threads = 0

        def _save_data(current_sensor, current_frame, current_data):
            nonlocal  current_threads
            current_threads += 1
            current_sensor.save_data(current_data, current_frame)
            current_threads -= 1

        missing_sensors = len(sensor_actors)
        results = []

        while True:
            frame = world.get_snapshot().frame
            try:
                sensor_data = sensor_queue.get(True, 2.0)
                if sensor_data[1] != frame:
                    continue
                missing_sensors -= 1
            except Empty:
                raise ValueError("A sensor took too long to send their data")

            sensor = sensor_data[0]
            frame_diff = sensor_data[1] - start_frame
            data = sensor_data[2]

            res = threading.Thread(target=_save_data, args=(sensor, frame_diff, data))
            results.append(res)
            res.start()

            if current_threads > self.config.simulation.save_threads:
                for res in results:
                    res.join()
                results = []

            if missing_sensors <= 0:
                break

    def _stop_world(self, client: Client, world: World):
        self.logger.debug("Setting world to unsync mode...")
        settings: WorldSettings = world.get_settings()
        settings.synchronous_mode = False

        self.logger.debug("Destroying actors...")
        client.apply_batch([carla.command.DestroyActor(x.id) for x in world.get_actors().filter('vehicle.*')])
        world.wait_for_tick()
        self.logger.debug("Finished destroying actors.")

        self.logger.debug("Stopping replayer...")
        client.stop_replayer(keep_actors=True)
        world.wait_for_tick()
        self.logger.debug("Finished stopping replayer.")

    def run(self):
        self.logger.info("Data generation started")

        self._carla_runner.start_carla()

        recordings = self.config.recordings if self.config.recordings else self._generate_recorder_info()
        for recorder_info in recordings:
            self.logger.info(f"Generating data for recording: {recorder_info['folder']}")

            recorder_name = os.path.split(recorder_info.folder)[-1]
            recorder_file = f"{os.path.join(recorder_info.folder, recorder_name)}.log"
            if not os.path.exists(recorder_file):
                self.logger.error(f"Recorder file {recorder_file} does not exist. Skipping...")
                continue

            client, _ = self._connect_to_server()
            recorder_data = self._get_recorder_data(recorder_file, client)
            destination_folder = self._create_destination_folder(recorder_info)

            self.logger.info(f"Setting up CARLA world for map: {recorder_data.map}...")
            world = self._setup_world(recorder_data, client)
            self.logger.debug("World set up.")

            self.logger.info(f"Setting up replay of recorded driving")
            start, end = self._get_recording_start_end(recorder_info, recorder_data)
            duration = end - start
            client.replay_file(os.path.abspath(recorder_file), start, duration, recorder_data.ego_vehicle['id'], False)
            world.tick()
            self.logger.debug(f"Replay of recorded driving setup.")

            self.logger.info(f"Setting up sensors")
            sensor_queue: Queue = Queue()
            ego_vehicle = world.get_actor(recorder_data.ego_vehicle['id'])
            sensor_actors = self._setup_sensors(world, ego_vehicle, sensor_queue)
            [world.tick() for i in range(10)]
            self.logger.debug(f"Sensors setup.")

            start_time = world.get_snapshot().timestamp.elapsed_seconds
            start_frame = world.get_snapshot().frame
            self.logger.info(f"Running the simulation replay from starting frame {start_frame}")

            while True:
                current_frame = world.get_snapshot().frame
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                completion = format(round(current_duration / duration * 100, 2), '3.2f')
                self._save_sensors(sensor_actors, world, sensor_queue, start_frame)
                self._save_ego_data(ego_vehicle=ego_vehicle, start_frame=start_frame, current_frame=current_frame,
                                    destination_folder=destination_folder)
                print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r", flush=True)
                world.tick()

            sensor_queue.queue.clear()
            self.logger.info(f"Stopping sensor actors")
            for sensor_actor in sensor_actors:
                sensor_actor.stop()
                sensor_actor.destroy()
            self.logger.debug(f"Stopping sensor actors")

            self.logger.debug(f"Stopping world")
            self._stop_world(client, world)

        self._carla_runner.terminate_carla(timeout=120)
        self.logger.info("Data generation finished")

    def _save_ego_data(self, ego_vehicle: carla.Vehicle, start_frame: int, current_frame: int, destination_folder: str):
        vehicle_control: carla.VehicleControl = ego_vehicle.get_control()
        control = vehicle_control.__dict__

        sensor_file = f"{destination_folder}/ego_log.csv"
        if not os.path.exists(sensor_file):
            with open(sensor_file, 'w') as data_file:
                header_txt = (f"frame,current_frame,t.x,t.y,t.z,t.yaw,t.pitch,t.roll,"
                              f"a.x,a.y,a.z,"
                              f"av.x,av.y,av.z,"
                              f"v.x,v.y,v.z\n")
                data_file.write(header_txt)
        with open(sensor_file, 'a') as data_file:
            t = ego_vehicle.get_transform()
            v = ego_vehicle.get_velocity()
            av = ego_vehicle.get_angular_velocity()
            a = ego_vehicle.get_acceleration()
            data_txt = (f"{current_frame-start_frame},{current_frame},{t.location.x},{t.location.y},{t.location.z},"
                        f"{t.rotation.yaw},{t.rotation.pitch},{t.rotation.roll},"
                        f"{a.x},{a.y},{a.z}"
                        f"{av.x},{av.y},{av.z},"
                        f"{v.x},{v.y},{v.z}\n")
            data_file.write(data_txt)
        pass


if __name__ == "__main__":
    logger = logging.root
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # create handler
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.DEBUG)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    file_handler = logging.FileHandler('carla-data-generator.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    config_path = "config.yaml"
    data_generator = CarlaDataGenerator(config_path)
    data_generator.run()
