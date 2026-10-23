/*
 * Hardware-in-the-Loop (HIL) Sensor Rig Firmware
 * Compatible with Arduino (Uno, Nano, Mega) and ESP32.
 *
 * Components:
 * - Servo Motor (PWM-controlled sweep)
 * - Ultrasonic Sensor (HC-SR04 / HC-SR04P)
 *
 * Operation:
 * - Non-blocking sweep routine between 30° and 150°.
 * - At each step, triggers the HC-SR04 to measure distance in centimeters.
 * - Streams JSON telemetry over Serial at 115200 baud for consumption by
 *   the terrain mapper Python serial parser (`serial_comms.py`).
 *
 * Serial Output Format (one JSON packet per line):
 *   {"angle": 45, "distance": 23.45}
 */

#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
  #include <ESP32Servo.h>
#else
  #include <Servo.h>
#endif

// ============================================================================
// Pin Configuration
// ============================================================================
#if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
  // Recommended pins for ESP32 DevKit
  constexpr int SERVO_PIN = 18;  // PWM capable pin
  constexpr int TRIG_PIN  = 5;   // Ultrasonic Trigger output
  constexpr int ECHO_PIN  = 19;  // Ultrasonic Echo input (Note: Use 3.3V divider if sensor is 5V)
#else
  // Standard Arduino (Uno, Nano, Mini) pins
  constexpr int SERVO_PIN = 9;   // PWM capable pin
  constexpr int TRIG_PIN  = 10;  // Ultrasonic Trigger output
  constexpr int ECHO_PIN  = 11;  // Ultrasonic Echo input
#endif

// ============================================================================
// Sweep & Timing Configuration
// ============================================================================
constexpr int MIN_ANGLE = 30;              // Minimum sweep angle (degrees)
constexpr int MAX_ANGLE = 150;             // Maximum sweep angle (degrees)
constexpr int ANGLE_STEP = 1;              // Step increment per interval
constexpr unsigned long STEP_INTERVAL_MS = 35; // Time between steps (ms)
constexpr unsigned long ECHO_TIMEOUT_US  = 30000; // ~5m max range timeout (30ms)

// ============================================================================
// Global State
// ============================================================================
Servo sweepServo;

int currentAngle = MIN_ANGLE;
int sweepDirection = 1; // 1 = forward (30 -> 150), -1 = backward (150 -> 30)
unsigned long lastStepTimestamp = 0;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Trigger HC-SR04 and compute distance in centimeters.
 * Returns -1.0 if echo timed out (out of range or disconnected).
 */
float readUltrasonicDistanceCm() {
  // Ensure clean LOW pulse before triggering
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  // Send 10µs HIGH pulse to trigger pin
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Measure echo pulse duration in microseconds
  unsigned long durationUs = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);

  if (durationUs == 0) {
    // Pulse timed out (no echo detected or out of range)
    return -1.0f;
  }

  // Speed of sound = 343 m/s = 0.0343 cm/µs
  // Distance = (duration / 2) * speed_of_sound
  return (float)(durationUs * 0.0343 / 2.0);
}

/**
 * Send JSON telemetry packet over Serial.
 * Format: {"angle": <int>, "distance": <float>}
 */
void sendTelemetry(int angle, float distanceCm) {
  Serial.print(F("{\"angle\": "));
  Serial.print(angle);
  Serial.print(F(", \"distance\": "));
  Serial.print(distanceCm, 2);
  Serial.println(F("}"));
}

// ============================================================================
// Setup & Main Loop
// ============================================================================

void setup() {
  // Initialize serial communication for HIL interface
  Serial.begin(115200);

  // Configure ultrasonic sensor pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  // Attach and position servo to starting angle
  #if defined(ESP32) || defined(ARDUINO_ARCH_ESP32)
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);
    sweepServo.setPeriodHertz(50); // Standard 50Hz servo
  #endif

  sweepServo.attach(SERVO_PIN);
  sweepServo.write(currentAngle);

  // Short warm-up delay for hardware stabilization
  delay(500);
  lastStepTimestamp = millis();
}

void loop() {
  unsigned long currentMillis = millis();

  // Non-blocking timer check for the sweep routine
  if (currentMillis - lastStepTimestamp >= STEP_INTERVAL_MS) {
    lastStepTimestamp = currentMillis;

    // 1. Move servo to current angle
    sweepServo.write(currentAngle);

    // 2. Measure distance at this angle
    float distance = readUltrasonicDistanceCm();

    // 3. Output clean JSON telemetry packet
    sendTelemetry(currentAngle, distance);

    // 4. Increment/decrement angle for next iteration
    currentAngle += sweepDirection * ANGLE_STEP;

    // 5. Reverse direction at limits
    if (currentAngle >= MAX_ANGLE) {
      currentAngle = MAX_ANGLE;
      sweepDirection = -1;
    } else if (currentAngle <= MIN_ANGLE) {
      currentAngle = MIN_ANGLE;
      sweepDirection = 1;
    }
  }

  // Main loop remains free for other non-blocking tasks if needed
}
