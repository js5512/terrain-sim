# Hardware-in-the-Loop (HIL) Firmware

This folder contains the microcontroller firmware for the physical scanning test rig (servo motor + ultrasonic sensor), providing real-time telemetry over USB/UART serial for the terrain simulator.

## Hardware Components
1. **Microcontroller**: ESP32 DevKit or Arduino (Uno, Nano, Mega).
2. **Servo Motor**: Standard micro-servo (e.g., SG90 or MG90S).
3. **Ultrasonic Sensor**: HC-SR04 or 3.3V-compatible HC-SR04P / RCWL-1601.

---

## Wiring & Pinout Guide

### Arduino (Uno / Nano)
| Component Pin | Arduino Pin | Notes |
| :--- | :--- | :--- |
| **Servo Signal (Orange/Yellow)** | **D9** | PWM output |
| **Servo VCC (Red)** | **5V** | External power recommended for larger servos |
| **Servo GND (Brown/Black)** | **GND** | Common ground with microcontroller |
| **HC-SR04 VCC** | **5V** | |
| **HC-SR04 GND** | **GND** | |
| **HC-SR04 TRIG** | **D10** | Digital output (Trigger pulse) |
| **HC-SR04 ECHO** | **D11** | Digital input (Echo pulse) |

### ESP32 (3.3V Logic)
| Component Pin | ESP32 Pin | Notes |
| :--- | :--- | :--- |
| **Servo Signal** | **GPIO 18** | PWM |
| **Servo VCC** | **VIN / 5V** | |
| **Servo GND** | **GND** | |
| **HC-SR04 VCC** | **5V** | |
| **HC-SR04 GND** | **GND** | |
| **HC-SR04 TRIG** | **GPIO 5** | 3.3V logic level output |
| **HC-SR04 ECHO** | **GPIO 19** | **Important:** If using standard 5V HC-SR04, use a voltage divider (e.g. 1kΩ / 2kΩ resistors) to drop the 5V echo pulse to ~3.3V before connecting to the ESP32 GPIO. |

---

## Library Requirements
- **Standard Arduino**: Uses the built-in `Servo.h` library.
- **ESP32**: Requires the `ESP32Servo` library (install via Arduino Library Manager or PlatformIO).

---

## Output Format
Data is output over USB Serial at **115200 baud** in line-delimited JSON format:
```json
{"angle": 45, "distance": 24.32}
```
- `angle`: Integer between 30 and 150 degrees.
- `distance`: Float measurement in centimeters (or `-1.00` if echo timed out).

---

## Integration with Simulator
To ingest live hardware readings from the serial port into the web simulation:
```bash
python -m terrain_mapper.web_sim --hardware --serial-port /dev/ttyUSB0 --baud-rate 115200
```
(On Windows, use `--serial-port COM3` or appropriate COM port).
