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
}

void runSequence(int seconds) {
  long totalSteps = (long)seconds * stepsPerSecond;

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

    if (cmd == 'P') {        // Paper (3s)
      runSequence(3);
    }
    else if (cmd == 'M') {   // Metal (6s)
      runSequence(6);
    }
    else if (cmd == 'L') {   // Plastic (9s)
      runSequence(9);
    }
    else if (cmd == 'T') {   // Trash (12s)
      runSequence(12);
    }

    Serial.println("Ready for next scan.");
  }
}