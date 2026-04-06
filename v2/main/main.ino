/*
 * Main file that gets compiled and uploaded 
 */

// #include "app_main.hpp"
// #include "calibration_main.hpp"
// #include "basic_receiver.hpp"
// #include "basic_sender.hpp"
// #include "basic_receiver2.hpp"
// #include "basic_sender2.hpp"
// #include "utils/imu.hpp"

// #define IS_ANCHOR // Comment out for Tag

#if defined (IS_ANCHOR)
#include "rtls/anchor.hpp"
#else
#include "rtls/tag.hpp"
#endif

void setup() {
  app_setup();
}

void loop() {
  app_loop();
}