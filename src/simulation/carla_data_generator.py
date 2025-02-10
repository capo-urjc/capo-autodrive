from __future__ import annotations

import logging

import os
import time
from threading import Lock
import carla

from src.simulation.tools import sim_logger
from src.simulation.tools.config import CarlaGeneratorConfig, read_config
from src.simulation.runner import CARLAServerRunner
from src.simulation.tools.replay_parser import get_recordings, Recording
from src.simulation.sensor import Sensor, SensorQueuedData


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class CarlaDataGenerator(metaclass=SingletonMeta):
    def __init__(self):
        self.logger = sim_logger.get_logger(self.__class__.__name__)
        self._initialized = False

        self._world: carla.World = None
        self._client: carla.Client = None
        self._sensors = []
        self._recordings: list[Recording] = []

    @property
    def world(self) -> carla.World:
        return self._world

    @property
    def client(self) -> carla.Client:
        if self._client is None:
            raise RuntimeError("Not connected to Carla server")

        return self._client

    def run(self, config_file: str):
        try:
            self._run(config_file)
        except Exception as e:
            self.logger.error("Error running simulation", exc_info=e)
        finally:
            if self._initialized:
                if self._carla_process:
                    self._carla_process.terminate_carla(120)


    def _run(self, config_file: str):
        self._init(config_file)

        # process all recordings
        for recording in self._recordings:
            self.logger.info(f"Collecting data for recording: {recording.name}")
            # get and create recording log folder if it does not exist
            recording_data_folder = os.path.join(self.config.files.destination_folder, recording.name)
            os.makedirs(recording_data_folder, exist_ok=True)

            # setup world for recording
            self.logger.info(f"Setting up CARLA world for map: {recording.recorder_data.map}...")
            self._setup_world(recording)
            self.logger.debug("World set up.")

            # start playing recording file
            self.logger.info(f"Setting up replay of recorded driving...")
            self.client.replay_file(recording.reply_file, recording.start_time, recording.duration,
                                    recording.ego_vehicle_id)
            self.world.tick()
            self.logger.debug(f"Replay of recorded driving setup.")

            # setup sensors
            hero: carla.Vehicle = self._world.get_actor(recording.ego_vehicle_id)
            sensor_data_provider: SensorQueuedData = SensorQueuedData(self.config.simulation.save_threads)
            for sensor_config in self.config.sensors:
                self.logger.debug(f"Setting up sensor {sensor_config.name}...")
                sensor = Sensor(recording_data_folder, sensor_config)
                sensor.setup(hero, self.client, sensor_data_provider)
                self._sensors.append(sensor)
                self.logger.debug(f"Sensor {sensor_config.name} setup.")

            for _ in range(10):
                self._world.tick()

            # run the simulation
            start_time = self._world.get_snapshot().timestamp.elapsed_seconds
            start_frame = self._world.get_snapshot().frame
            total_duration = recording.duration

            self.logger.info(f"Running the simulation replay at {start_time} seconds from starting frame {start_frame}")
            while True:
                current_time = self._world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                simulation_completed_pct = round(current_duration / total_duration  * 100, 2)
                current_frame = self._world.get_snapshot().frame

                # The "or" section is for debugging purposes only
                #  or current_frame - start_frame >= 20
                if current_duration >= total_duration:
                    print(f">>>>> Running recorded simulation frame [{current_frame - start_frame}]: "
                          f"100.00%  completed  <<<<<", flush=True)
                    break

                sensor_data_provider.save_sensors(current_frame, start_frame)
                self._save_ego_data(ego_vehicle=hero, start_frame=start_frame, current_frame=current_frame,
                                    destination_folder=recording_data_folder)

                print(f">>>>>  Running recorded simulation frame [{current_frame - start_frame}]:"
                      f" {simulation_completed_pct:3.2f}%  completed  <<<<<", end="\r", flush=True)

                self._world.tick()

            self._reset_world()

    def _init(self, config_file: str):
        # read configuration data
        self.config = read_config(config_file=config_file)

        # start carla server
        self._carla_process = CARLAServerRunner(port=self.config.server.port)
        self._carla_process.start_carla()

        # connect to carla server
        self._connect_to_server()

        # load recordings information
        self._recordings = get_recordings(self.config, self.client)

        # initialized suceeded
        self._initialized = True

    def _connect_to_server(self, timeout: int = 20, max_retries: int = 5, retry_interval: int = 5):
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Attempt {attempt} to connect to CARLA server...")
                self._client = carla.Client(self.config.server.host, self.config.server.port)
                self._client.set_timeout(timeout)
                self.logger.info("Successfully connected to CARLA server.")
                break
            except Exception as e:
                self.logger.debug(f"Connection attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    self.logger.debug(f"Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                else:
                    raise RuntimeError("Exceeded maximum retries. CARLA server is not ready.")

    def _setup_world(self, recording: Recording):
        self.client.set_timeout(240)

        # Get the world
        self._world = self.client.load_world(recording.recorder_data.map)
        self._world.tick()

        # Set world weather
        weather = carla.WeatherParameters(**self.config.weather)
        self._world.set_weather(weather)

        # Change fps and sync mode
        world_settings: carla.WorldSettings = self._world.get_settings()
        world_settings.fixed_delta_seconds = 1 / self.config.simulation.fps
        world_settings.synchronous_mode = True
        self._world.apply_settings(world_settings)
        self._world.tick()

    def _reset_world(self):
        self.logger.info("Resetting world.")
        world_settings: carla.WorldSettings = self._world.get_settings()
        world_settings.fixed_delta_seconds = None
        world_settings.synchronous_mode = False
        self._world.apply_settings(world_settings)
        self._world.wait_for_tick()

        self._client.stop_replayer(False)

        # Delete sensors
        for i in range(len(self._sensors)):
            self._sensors[i].destroy()
        self._sensors.clear()

        all_vehicles = self._world.get_actors().filter('vehicle.*')
        for i in range(0, len(all_vehicles)):
            all_vehicles[i].destroy()

        self._world.wait_for_tick()

        # self._client.apply_batch([carla.command.DestroyActor(x.id) for x in all_vehicles])
        self.logger.debug("Finished world reset.")

    def _save_ego_data(self, ego_vehicle, start_frame, current_frame, destination_folder):
        vehicle_control: carla.VehicleControl = ego_vehicle.get_control()
        control = vehicle_control.__dict__

        ego_file = f"{destination_folder}/ego_log.csv"
        
        if not os.path.exists(ego_file):
            with open(ego_file, 'w') as data_file:
                header_txt = (f"frame,current_frame,t.x,t.y,t.z,t.yaw,t.pitch,t.roll,"
                              f"a.x,a.y,a.z,"
                              f"av.x,av.y,av.z,"
                              f"v.x,v.y,v.z\n")
                data_file.write(header_txt)
                
        with open(ego_file, 'a') as data_file:
            t = ego_vehicle.get_transform()
            v = ego_vehicle.get_velocity()
            av = ego_vehicle.get_angular_velocity()
            a = ego_vehicle.get_acceleration()
            data_txt = (f"{current_frame - start_frame},{current_frame},{t.location.x},{t.location.y},{t.location.z},"
                        f"{t.rotation.yaw},{t.rotation.pitch},{t.rotation.roll},"
                        f"{a.x},{a.y},{a.z}"
                        f"{av.x},{av.y},{av.z},"
                        f"{v.x},{v.y},{v.z}\n")
            data_file.write(data_txt)

if __name__ == "__main__":
    import sys
    import click

    logger = logging.root
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(sim_logger.VERBOSE_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file_handler = logging.FileHandler('carla-data-generator.log')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    @click.command()
    @click.option("-c", "--config", 'config_file', required=True,
                  type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=str),
                  help="Configuration file")
    def main(config_file: str):
        data_generator = CarlaDataGenerator()
        data_generator.run(config_file=config_file)

    main()