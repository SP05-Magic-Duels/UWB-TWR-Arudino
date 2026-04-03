// TAG — 1 anchor, 2 tag MVP
// Prints distance to anchor and sends via ESP-NOW on every range measurement.
// Uses millis() slots to prevent both tags transmitting simultaneously.
//
// FLASH INSTRUCTIONS:
//   Tag 1: leave   #define FLASH_TAG1   uncommented
//   Tag 2: comment #define FLASH_TAG1,  uncomment #define FLASH_TAG2
//
// Power both tags at the same time to keep slots aligned.

#include <SPI.h>
#include "DW1000Ranging.h"
#include "DW1000.h"

#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>

// ==========================================
// TAG SELECTION
// ==========================================
#define FLASH_TAG1
// #define FLASH_TAG2
// ==========================================

// ── Slot gate ─────────────────────────────
#define SLOT_MS 100  // ms per tag (total cycle = 200ms)

#if defined(FLASH_TAG1)
  #define MY_SLOT_START 0
#elif defined(FLASH_TAG2)
  #define MY_SLOT_START SLOT_MS
#endif

inline bool mySlot() {
  unsigned long phase = millis() % (SLOT_MS * 2UL);
  return (phase >= (unsigned long)MY_SLOT_START &&
          phase <  (unsigned long)(MY_SLOT_START + SLOT_MS));
}
// ──────────────────────────────────────────

// ── UWB address ───────────────────────────
#if defined(FLASH_TAG1)
  char tag_addr[] = "7D:00:22:EA:82:60:3B:9C";
#elif defined(FLASH_TAG2)
  char tag_addr[] = "7E:00:22:EA:82:60:3B:9C";
#endif
// ──────────────────────────────────────────

// ── ESP-NOW ───────────────────────────────
// RCVR: C8:F0:9E:F1:AB:5C
uint8_t targetAddress[] = {0xC8, 0xF0, 0x9E, 0xF1, 0xAB, 0x5C};

typedef struct struct_message {
  float distance;
} struct_message;

struct_message txData;
esp_now_peer_info_t peerInfo = {};

void ESPNOW_Init() {
  WiFi.mode(WIFI_STA);
  WiFi.begin();
  while (WiFi.macAddress() == "00:00:00:00:00:00") { delay(10); }
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  Serial.print("MAC: "); Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) { Serial.println("ESP-NOW init failed"); return; }
  memcpy(peerInfo.peer_addr, targetAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);
}
// ──────────────────────────────────────────

// ── UWB hardware ─────────────────────────
#define SPI_SCK  18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS    4

const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS  = 4;
// ──────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(1000);

  ESPNOW_Init();

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);
  DW1000Ranging.startAsTag(tag_addr, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);

  #if defined(FLASH_TAG1)
    Serial.println("Tag 1 ready — slot 0-99ms");
  #elif defined(FLASH_TAG2)
    Serial.println("Tag 2 ready — slot 100-199ms");
  #endif
}

void loop() {
  if (mySlot()) {
    DW1000Ranging.loop();
  }
}

void newRange() {
  float range = DW1000Ranging.getDistantDevice()->getRange();

  #if defined(FLASH_TAG1)
    Serial.print("T1 dist: ");
  #elif defined(FLASH_TAG2)
    Serial.print("T2 dist: ");
  #endif
  Serial.println(range);

  txData.distance = range;
  esp_now_send(targetAddress, (uint8_t *)&txData, sizeof(txData));
}

void newDevice(DW1000Device *device) {
  Serial.print("Device added: ");
  Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device *device) {
  Serial.print("Delete inactive device: ");
  Serial.println(device->getShortAddress(), HEX);
}
