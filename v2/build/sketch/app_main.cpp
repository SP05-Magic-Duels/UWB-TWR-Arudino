#line 1 "/home/choafe/Arduino/UWB-TWR-Arudino/v2/main/app_main.cpp"
#include <Arduino.h>
#include "app_main.h"

// The setup function runs once when you press reset or power the board
void app_setup() {
  // Initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
}

// The loop function runs over and over again forever
void app_loop() {
  digitalWrite(LED_BUILTIN, HIGH);  // Turn the LED on
  delay(1000);                      // Wait for a second
  digitalWrite(LED_BUILTIN, LOW);   // Turn the LED off
  delay(1000);                      // Wait for a second
}
