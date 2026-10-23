import io
import time
from unittest.mock import MagicMock, patch
from terrain_mapper.serial_comms import SerialTelemetryParser


def test_serial_telemetry_parser_parsing():
    """Verify that the parser correctly extracts angle and distance from JSON packets."""
    parser = SerialTelemetryParser(port="MOCK_PORT", baudrate=115200)

    # Simulate serial line reading
    mock_serial = MagicMock()
    mock_serial.is_open = True
    mock_serial.readline.side_effect = [
        b'{"angle": 45, "distance": 23.45}\n',
        b'corrupted json line\n',
        b'{"angle": 90, "distance": 12.00}\n',
    ]

    with patch("serial.Serial", return_value=mock_serial):
        parser.start()
        # Give thread a brief moment to process the mock reads
        time.sleep(0.05)
        parser.stop()

    latest = parser.latest_telemetry
    assert latest is not None
    assert latest["angle"] == 90.0
    assert latest["distance"] == 12.00
