import logging
import os
import signal
import socket
import subprocess
import time

import psutil
from threading import Thread


class CARLAServerRunner:
    def __init__(self, port: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._carla_process = None
        self._port = port

    def start_carla(self):
        self.logger.debug(f"Starting CARLA simulation server on port {self._port}...")
        carla_root = os.getenv("CARLA_ROOT")
        bin_file = f"{carla_root}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
        carla_port = f"-carla-port={self._port}"
        command = [bin_file, "CarlaUE4", "-quality-level=Epic", "-world-port=2000", "-resx=800", "-resy=600",
                   "-opengl", "-RenderOffScreen", carla_port]

        self._carla_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                               cwd=carla_root)

        # Set pipes to non-blocking mode
        os.set_blocking(self._carla_process.stdout.fileno(), False)
        os.set_blocking(self._carla_process.stderr.fileno(), False)

        # Start a thread to continuously read CARLA server output and log it
        Thread(target=self._log_carla_output, daemon=True).start()
        self._wait_for_port()

    def _log_carla_output(self):
        """
        Log the output of the CARLA server process to the logger with a prefix to distinguish CARLA messages.
        """
        while True:
            try:
                stdout_line = self._carla_process.stdout.readline()
                if stdout_line:
                    self.logger.info(f"[CARLA STDOUT] {stdout_line.strip()}")

                stderr_line = self._carla_process.stderr.readline()
                if stderr_line:
                    self.logger.error(f"[CARLA STDERR] {stderr_line.strip()}")

                # Check if the process has ended
                if self._carla_process.poll() is not None:
                    break

            except Exception as e:
                self.logger.error(f"Error reading CARLA output: {e}")

            time.sleep(0.1)  # Prevent tight loop, reduce CPU usage

    def _wait_for_port(self, timeout: int = 30, poll_interval: int = 1):
        host = "127.0.0.1"
        ports = [self._port, self._port + 1, self._port + 2]

        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ports_ready = True
            for port in ports:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.settimeout(1.0)
                        s.connect((host, port))
                    except (socket.timeout, ConnectionRefusedError):
                        self.logger.debug(f"CARLA port {port} not ready yet")
                        all_ports_ready = False
                        break
            if all_ports_ready:
                self.logger.debug(f"All CARLA ports ready on {host}: {ports}")
                return
            time.sleep(poll_interval)
        raise RuntimeError(f"Timeout waiting for CARLA ports on {host}: {ports}")

    def _process_is_alive(self):
        return self._carla_process.poll() is None

    def _terminate_process_tree(self, timeout=30):
        """
        Terminate the given process and all its children gracefully, then forcefully if needed.
        """
        try:
            parent = psutil.Process(self._carla_process.pid)
            children = parent.children(recursive=True)

            self.logger.info(f"Sending SIGINT to CARLA process group {parent.pid}...")
            os.kill(self._carla_process.pid, signal.SIGINT)
            gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

            if alive:
                # Step 3: If some processes are still alive, escalate to SIGKILL
                self.logger.warning(
                    f"Processes {', '.join(str(p.pid) for p in alive)} did not terminate gracefully. Force killing...")
                for p in alive:
                    p.kill()

                # Step 4: Wait again for the remaining processes to be killed
                gone, alive = psutil.wait_procs(alive, timeout=10)

                if alive:
                    self.logger.error(f"Failed to forcefully kill processes: {', '.join(str(p.pid) for p in alive)}")
                else:
                    self.logger.info("All CARLA processes have been killed.")
            else:
                self.logger.info("CARLA server terminated gracefully.")

        except psutil.NoSuchProcess:
            self.logger.debug(f"Process {self._carla_process.pid} already terminated.")

    def terminate_carla(self, timeout: int):
        """
        Terminate the CARLA server gracefully, escalating to a forced kill if necessary.
        """
        self.logger.info("Attempting to stop CARLA server...")
        if self._process_is_alive():
            self._terminate_process_tree(timeout=timeout)
        else:
            self.logger.info("CARLA server is no longer running.")
