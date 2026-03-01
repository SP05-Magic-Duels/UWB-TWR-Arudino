//anchor #1 setup
// Standard anchor — no changes needed for 2-tag system.
// Flash all 4 anchors with this sketch, changing anchor_addr and Adelay each time.

#include <SPI.h>
#include "DW1000Ranging.h"
#include "DW1000.h"

char anchor_addr[] = "81:00:5B:D5:A9:9A:E2:9C"; //#1  ← change per anchor: 81, 82, 83, 84

uint16_t Adelay = 16630; // #1=16630, #2=16610, #3=16607, #4=16580

float dist_m = (285 - 1.75) * 0.0254;

#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4

const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS  = 4;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Anchor config and start");
  Serial.print("Antenna delay "); Serial.println(Adelay);
  Serial.print("Calibration distance "); Serial.println(dist_m);

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  DW1000.setAntennaDelay(Adelay);

  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);

  DW1000Ranging.startAsAnchor(anchor_addr, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);
}

void loop() {
  DW1000Ranging.loop();
}

void newRange() {
  Serial.print(DW1000Ranging.getDistantDevice()->getShortAddress(), HEX);
  Serial.print(", ");
  Serial.println(DW1000Ranging.getDistantDevice()->getRange());
}

void newDevice(DW1000Device *device) {
  Serial.print("Device added: ");
  Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device *device) {
  Serial.print("Delete inactive device: ");
  Serial.println(device->getShortAddress(), HEX);
}
