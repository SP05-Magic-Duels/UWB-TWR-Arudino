#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>

// #define FLASH_TRANMITTER_RECEIVER // Uncomment to flash

// ==========================================
// CONFIGURATION: Set the role of this device
// ==========================================
#define ROLE_TRANSMITTER    // Options: ROLE_TRANSMITTER, ROLE_RECEIVER
// ==========================================

// MAC ADDRESSES
// TAG1: E0:5A:1B:93:EF:14
// TAG2: ?
// RCVR: C8:F0:9E:F1:AB:5C

#if defined(ROLE_TRANSMITTER)
  // If I am the Transmitter, I must target the RECEIVER'S MAC
  uint8_t targetAddress[] = {0xC8, 0xF0, 0x9E, 0xF1, 0xAB, 0x5C};
#else
  // If I am the Receiver, I target the TAG'S MAC (only needed for two-way)
  uint8_t targetAddress[] = {0xE0, 0x5A, 0x1B, 0x93, 0xEF, 0x14};
#endif

typedef struct struct_message {
    float x;
    float y;
    float z;
} struct_message;

struct_message data;
esp_now_peer_info_t ESPNOW_peerInfo = {};

// Updated for ESP32 Core 3.x API
void ESPNOW_OnDataSent(const wifi_tx_info_t *txInfo, esp_now_send_status_t status) {
  Serial.print("Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");
}

// Updated for ESP32 Core 3.x API
void ESPNOW_OnDataRecv(const esp_now_recv_info_t *recvInfo, const uint8_t *incomingData, int len) {
  memcpy(&data, incomingData, sizeof(data));
  Serial.printf("From: %02X:%02X:%02X | XYZ: %.2f, %.2f, %.2f\n", 
                recvInfo->src_addr[0], recvInfo->src_addr[3], recvInfo->src_addr[5],
                data.x, data.y, data.z);
}

#ifdef FLASH_TRANMITTER_RECEIVER

void setup() {
  Serial.begin(115200);
  delay(1000); 

  WiFi.mode(WIFI_STA);
  WiFi.STA.begin(); // Critical for waking up the hardware radio
  
  // Force same channel
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

  Serial.print("My MAC: ");
  Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  #if defined(ROLE_TRANSMITTER)
    esp_now_register_send_cb(esp_now_send_cb_t(ESPNOW_OnDataSent));
    
    memcpy(ESPNOW_peerInfo.peer_addr, targetAddress, 6);
    ESPNOW_peerInfo.channel = 0; // Use current channel
    ESPNOW_peerInfo.encrypt = false;
    
    if (esp_now_add_peer(&ESPNOW_peerInfo) != ESP_OK) {
      Serial.println("Failed to add peer");
    }
  #else
    esp_now_register_recv_cb(esp_now_recv_cb_t(ESPNOW_OnDataRecv));
  #endif
}

void loop() {
  #if defined(ROLE_TRANSMITTER)
    data.x = 1.5;
    data.y = 2.8;
    data.z = 0.5;

    esp_now_send(targetAddress, (uint8_t *) &data, sizeof(data));
    delay(2000); 
  #endif
}

#endif // End of FLASH_TRANMITTER_RECEIVER
