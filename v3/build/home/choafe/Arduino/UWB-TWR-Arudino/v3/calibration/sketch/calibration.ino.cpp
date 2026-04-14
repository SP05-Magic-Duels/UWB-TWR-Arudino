#include <Arduino.h>
#line 1 "/home/choafe/Arduino/UWB-TWR-Arudino/v3/calibration/calibration.ino"
/*
 * Main file that gets compiled and uploaded 
 */

#define IS_ANCHOR // Comment out for Tag

#if defined (IS_ANCHOR)
#include "responder.hpp"
#else
#include "initiator.hpp"
#endif

#line 13 "/home/choafe/Arduino/UWB-TWR-Arudino/v3/calibration/calibration.ino"
void setup();
#line 17 "/home/choafe/Arduino/UWB-TWR-Arudino/v3/calibration/calibration.ino"
void loop();
#line 13 "/home/choafe/Arduino/UWB-TWR-Arudino/v3/calibration/calibration.ino"
void setup() {
  app_setup();
}

void loop() {
  app_loop();
}
