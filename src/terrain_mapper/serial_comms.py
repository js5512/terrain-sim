import json
import logging
import threading
import time
from typing import Optional, Dict

try:
    import serial
except ImportError:
    serial = None

logger = logging.getLogger(__name__)

class SerialTelemetryParser:
    """
    Handles Hardware-in-the-Loop (HIL) serial communication and 
    JSON telemetry parsing for ultrasonic and servo data.
    """
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self._latest_telemetry: Optional[Dict[str, float]] = None
        self._lock = threading.Lock()

    def start(self):
        if serial is None:
            raise ImportError("pyserial is not installed. Please install it to use hardware mode.")
            
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            logger.info(f"Started HIL serial connection on {self.port} at {self.baudrate}")
        except serial.SerialException as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            raise

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Serial connection closed.")

    def _read_loop(self):
        while self.running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line:
                    data = json.loads(line)
                    # Expected JSON format: {"distance": 15.5, "angle": 45}
                    if "distance" in data and "angle" in data:
                        with self._lock:
                            self._latest_telemetry = {
                                "distance": float(data["distance"]),
                                "angle": float(data["angle"])
                            }
            except json.JSONDecodeError:
                # Ignore malformed JSON packets that might occur mid-transmission
                pass
            except serial.SerialException:
                logger.error("Serial connection lost.")
                self.running = False
            except Exception as e:
                logger.debug(f"Error parsing serial telemetry: {e}")
            
            # Small sleep to yield to other threads if spinning fast
            time.sleep(0.001)

    @property
    def latest_telemetry(self) -> Optional[Dict[str, float]]:
        """Returns the most recent parsed telemetry as a dictionary, or None if no data."""
        with self._lock:
            return self._latest_telemetry.copy() if self._latest_telemetry else None
