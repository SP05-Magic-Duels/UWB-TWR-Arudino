/*
 * Main file that gets compiled and uploaded 
 */

#define IS_ANCHOR // Comment out for Tag

#if defined (IS_ANCHOR)
#include "responder.hpp"
#else
#include "initiator.hpp"
#endif

void setup() {
  app_setup();
}

void loop() {
  app_loop();
}