// ANCHOR with TRIGGER BEACON
// Broadcasts a single byte 0xAA repeatedly as a ranging trigger.
// Tags self-differentiate by delay: Tag 1 ranges immediately, Tag 2 waits 20ms.
//
// Only flash ONE anchor with this sketch. The other three keep the original code.

#include <SPI.h>
#include "DW1000Ranging.h"
#include "DW1000.h"

#define TRIGGER_INTERVAL 60   // ms between triggers (must be > 2x TAG_SLOT_MS, so > 40ms)
#define TRIGGER_BYTE     0xAA

char anchor_addr[] = "84:00:5B:D5:A9:9A:E2:9C"; // change per anchor
uint16_t Adelay = 16660;

#define SPI_SCK  18
#define SPI_MISO 19
#define SPI_MOSI 23

const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS  = 4;

unsigned long lastTriggerTime = 0;

void sendTrigger() {
  byte msg[1] = { TRIGGER_BYTE };
  DW1000.newTransmit();
  DW1000.setDefaults();
  DW1000.setData(msg, 1);
  DW1000.startTransmit();
  Serial.println("TRIGGER sent");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  DW1000.setAntennaDelay(Adelay);

  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);

  DW1000Ranging.startAsAnchor(anchor_addr, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);

  sendTrigger();
  lastTriggerTime = millis();
}

void loop() {
  DW1000Ranging.loop();

  if (millis() - lastTriggerTime >= TRIGGER_INTERVAL) {
    lastTriggerTime = millis();
    sendTrigger();
  }
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
