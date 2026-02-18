#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>

#define FLASH_TRANMITTER_RECEIVER // Uncomment to flash

// ==========================================
// CONFIGURATION
// ==========================================
#define ROLE_RECEIVER    // Options: ROLE_TRANSMITTER, ROLE_RECEIVER
// Note that the receiver is always on always listen mode
// ==========================================

// MAC ADDRESSES
// TAG1: E0:5A:1B:93:EF:14
// TAG2: ?
// RCVR: C8:F0:9E:F1:AB:5C

#if defined(ROLE_TRANSMITTER)
// Tags target the Receiver
uint8_t targetAddress[] = {0xC8, 0xF0, 0x9E, 0xF1, 0xAB, 0x5C};
#endif

typedef struct struct_message {
  float x;
  float y;
  float z;
} struct_message;

struct_message data;

// Updated for ESP32 Core 3.x API
void ESPNOW_OnDataSent(const wifi_tx_info_t *txInfo, esp_now_send_status_t status) {
  if (status != ESP_NOW_SEND_SUCCESS) {
    Serial.println("Send Status: Fail");
  }
}

// Updated for ESP32 Core 3.x API to handle multiple sources
void ESPNOW_OnDataRecv(const esp_now_recv_info_t *recvInfo, const uint8_t *incomingData, int len) {
  memcpy(&data, incomingData, sizeof(data));

  // Identify which Tag sent the data based on MAC
  Serial.print("Tag [");
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", recvInfo->src_addr[i]);
    if (i < 5) Serial.print(":");
  }
  Serial.printf("] | XYZ: %.2f, %.2f, %.2f\n", data.x, data.y, data.z);
}

#ifdef FLASH_TRANMITTER_RECEIVER

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.STA.begin(); // Critical for waking up the hardware radio

  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE); // Force same channel

  Serial.print("My MAC: ");
  Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  #if defined(ROLE_TRANSMITTER)
  esp_now_register_send_cb(esp_now_register_send_cb_t(ESPNOW_OnDataSent));

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, targetAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
  }
  #else
  // Receivers just register the callback; they don't need to add peers to listen
  esp_now_register_recv_cb(esp_now_recv_cb_t(ESPNOW_OnDataRecv));
  #endif
}

void loop() {
  #if defined(ROLE_TRANSMITTER)
  // In your actual project, these come from your trilateration function
  data.x = 1.5;
  data.y = 2.8;
  data.z = 0.5;

  esp_now_send(targetAddress, (uint8_t *) &data, sizeof(data));
  delay(2000);
  #endif
}

#endif
