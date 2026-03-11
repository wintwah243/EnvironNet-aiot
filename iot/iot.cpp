// This is a code example to run on arduino ide.
// Run this Arduino code in Arduino IDE for fully working program.

#include <Servo.h>

Servo myServo;
const int dirPin = 2;
const int stepPin = 3;

const int stepsPerSecond = 250;

void setup() {
  myServo.attach(8);
  myServo.write(0);

  pinMode(dirPin, OUTPUT);
  pinMode(stepPin, OUTPUT);

  Serial.begin(9600);
  Serial.println("System Ready. Send P, M, L, or T.");
}

void runSequence(float seconds) {
  long totalSteps = (long)(seconds * stepsPerSecond);

  digitalWrite(dirPin, LOW);
  for (long x = 0; x < totalSteps; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(2000);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(2000);
  }

  delay(500);

  myServo.write(90);
  delay(2000);

  myServo.write(0);
  delay(1000);

  digitalWrite(dirPin, HIGH);
  for (long x = 0; x < totalSteps; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(2000);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(2000);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'P') {
      Serial.println("Paper Detected: 0s Move");
      runSequence(0);
    }
    else if (cmd == 'M') {
      Serial.println("Metal Detected: 0.4s Move");
      runSequence(0.4);
    }
    else if (cmd == 'L') {
      Serial.println("Plastic Detected: 1.2s Move");
      runSequence(1.2);
    }
    else if (cmd == 'T') {
      Serial.println("Trash Detected: 1.8s Move");
      runSequence(1.8);
    }

    Serial.println("Ready for next scan.");
  }
}