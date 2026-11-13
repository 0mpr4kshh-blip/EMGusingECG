// ECG acquisition: reads AD8232 output on A0 at 250 samples/s
// and sends one integer per line over USB serial.
// A line containing -1 means "lead off" (electrode not connected).

const int ECG_PIN = A0;
const int LO_PLUS = 10;                        // AD8232 LO+
const int LO_MINUS = 11;                       // AD8232 LO-
const unsigned long SAMPLE_INTERVAL_US = 4000; // 4 ms = 250 Hz

unsigned long nextSampleTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  nextSampleTime = micros();
}

void loop() {
  unsigned long now = micros();
  // (long) cast keeps this correct when micros() wraps around
  if ((long)(now - nextSampleTime) >= 0) {
    nextSampleTime += SAMPLE_INTERVAL_US;   // schedule next sample
    if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
      Serial.println(-1);                   // lead off
    } else {
      Serial.println(analogRead(ECG_PIN));  // 0 to 1023
    }
  }
}
