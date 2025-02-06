from __future__ import annotations

import logging
import os
import pathlib
import re
from datetime import datetime

import carla
from typing import List

from src.simulation.tools.config import CarlaGeneratorConfig


def _generate_recorder_info(base_folder):
    folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    recorder_info = []

    for folder in folders:
        full_folder = os.path.join(base_folder, folder)

        log_file = next(
            (file.name for file in pathlib.Path(full_folder).iterdir() if file.is_file() and file.suffix == ".log"),
            None)

        if log_file is None:
            continue

        name = os.path.splitext(log_file)[0]
        recorder_info.append({
            'folder': full_folder,
            'name': name,
            'start_time': 0,
            'duration': 0
        })

    return recorder_info

def get_recordings(config: CarlaGeneratorConfig, client: carla.Client) -> List[Recording]:
    config_recordings = config.recordings
    recordings: List[Recording] = []

    if config_recordings is None:
        config_recordings = _generate_recorder_info(config.files.recordings_folder)

    for config_recording in config_recordings:
        recording = Recording(config_recording, client)
        recordings.append(recording)

    return recordings

class Recording(object):
    def __init__(self, config: dict, client: carla.Client):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.folder = config['folder']

        self._recorder_file = None
        path = pathlib.Path(self.folder)
        for file in path.glob("*.log"):
            self._recorder_file = file

        if not os.path.exists(self._recorder_file):
            raise FileNotFoundError(f"Recorder file not found in {self.folder}")

        recorder_str = client.show_recorder_file_info(os.path.abspath(self._recorder_file), False)
        self.recorder_data = ReplayParser(recorder_str)

        # get rest of configuration
        self._name = config.get('name')

        self._start_time = config.get('start_time', 0)
        if self._start_time >= self.recorder_data.duration:
            self.logger.warning("Found a start point that exceeds recorder length, adjusting start_point to 0")

        self._duration = config.get('duration', 0)
        if self._duration == 0:
            self._duration = self.recorder_data.duration
        elif self._start_time + self._duration > self.recorder_data.duration:
            self.logger.warning("Found a duration that exceeds recorder length, adjusting it to recorder length")
            self._duration = self.recorder_data.duration - self._start_time

    @property
    def recorder_file(self):
        return self._recorder_file

    @property
    def reply_file(self):
        return os.path.abspath(self._recorder_file)

    @property
    def name(self) -> str:
        return self._name

    @property
    def start_time(self) -> int:
        return self._start_time

    def end_time(self):
        return self._start_time + self._duration

    @property
    def duration(self) -> int:
        return self._duration

    @property
    def ego_vehicle_id(self) -> int:
        return self.recorder_data.ego_vehicle['id']



class ReplayParser:
    def __init__(self, replay_str: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._replay_str = replay_str
        self._replay_lines = replay_str.split('\n')
        self.version = None
        self.date = None
        self.duration = None
        self.total_frames = None
        self.map = None
        self.frames = []
        self._parse_replay_str()

    def _parse_replay_str(self):
        current_frame = None        
        
        for i, line in enumerate(self._replay_lines):
            line = line.strip()
            if line.startswith("Version:"):
                self.version = self._parse_version(line)
            elif line.startswith('Date:'):
                self.date = self._parse_date(line)
            elif line.startswith("Duration:"):
                self.duration = self._parse_duration(line)
            elif line.startswith('Frames:'):
                self.total_frames = self._parse_frames(line)
            elif line.startswith('Map:'):
                self.map = self._parse_map(line)
            elif line.startswith('Frame'):                
                if current_frame:
                    self.frames.append(current_frame)
                current_frame = self._parse_frame(line)
            elif line.startswith("Create") or line.startswith("Destroy"):
                if current_frame is not None:
                    current_frame["actions"].append(self._parse_action(line, self._replay_lines[i+1:]))

        if current_frame:
            self.frames.append(current_frame)

    @staticmethod
    def _parse_var(line, var: str, suff: str = None):
        reg_exp = fr"{var}: (.+) {suff}" if suff is not None else fr"{var}: (.+)"
        match = re.search(reg_exp, line)
        return match.group(1) if match else None

    def _parse_date(self, line):
        date_str = self._parse_var(line, "Date")
        if date_str:
            try:
                return datetime.strptime(date_str, "%m/%d/%y %H:%M:%S")
            except ValueError:
                return date_str  # Return as string if parsing fails
        return None

    def _parse_version(self, line):
        return self._parse_var(line, "Version")

    def _parse_duration(self, line):
        duration = self._parse_var(line, "Duration", 'seconds')
        return float(duration) if duration else None

    def _parse_frames(self, line):
        total_frames = self._parse_var(line, "Frames")
        return int(total_frames) if total_frames else None

    def _parse_map(self, line):
        return self._parse_var(line, 'Map')

    def _parse_frame(self, line):
        match = re.search(r"Frame (\d+) at ([\d\.]+) seconds", line)
        if match:
            return {
                "frame_id": int(match.group(1)),
                "time": float(match.group(2)),
                "actions": []
            }
        return None

    def _parse_action(self, line, following_lines):
        if line.startswith("Create"):
            match = re.search(r"Create (\d+): (.+) \((\d+)\) at \(([^)]+)\)", line)
            if match:
                action_data = {
                    "action": "Create",
                    "id": int(match.group(1)),
                    "type": match.group(2),
                    "parent_id": int(match.group(3)),
                    "location": tuple(map(float, match.group(4).split(', ')))
                }
                # Check if additional properties exist for vehicles
                if "vehicle" in action_data["type"]:
                    vehicle_properties = self._parse_vehicle_properties(following_lines)
                    action_data.update(vehicle_properties)
                return action_data
        elif line.startswith("Destroy"):
            match = re.search(r"Destroy (\d+)", line)
            if match:
                return {
                    "action": "Destroy",
                    "id": int(match.group(1))
                }
        return None

    def _parse_vehicle_properties(self, following_lines):
        properties = {}
        for prop_line in following_lines:
            prop_line = prop_line.strip()
            if not prop_line or prop_line.startswith("Frame") or prop_line.startswith("Create") or prop_line.startswith("Destroy"):
                break
            if "=" in prop_line:
                key, value = prop_line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Attempt to parse value to correct type
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "," in value:
                    value = tuple(map(int, value.split(",")))
                properties[key] = value
        return properties

    @property
    def ego_vehicle(self) -> dict:
        for frame in self.frames:
            for action in frame["actions"]:
                if action["action"] == "Create" and action.get("role_name") == "hero":
                    return action
        return None