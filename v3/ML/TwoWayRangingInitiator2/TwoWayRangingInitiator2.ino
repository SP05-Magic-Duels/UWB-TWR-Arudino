#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgTime.hpp>
#include <DW1000NgConstants.hpp>
#include <WiFi.h>
#include <esp_now.h>

#define NUM_ANCHORS 8

const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS = 4;

#define POLL 0
#define POLL_ACK 1
#define RANGE 2
#define RANGE_REPORT 3
#define RANGE_FAILED 255

volatile byte expectedMsgId = POLL_ACK;
volatile boolean sentAck = false;
volatile boolean receivedAck = false;

uint64_t timePollSent;
uint64_t timePollAckReceived;
uint64_t timeRangeSent;

#define LEN_DATA 20
byte data[LEN_DATA];

uint32_t lastActivity;
uint32_t resetPeriod = 30;
uint16_t replyDelayTimeUS = 3000;

enum TagState { IDLE, POLLING };
TagState currentState = IDLE;
uint32_t idleTimer = 0;
const uint32_t IDLE_DELAY_MS = 10;

uint8_t current_anchor = 1;
float distances[NUM_ANCHORS] = {0};
float rx_powers[NUM_ANCHORS] = {0};
float fp_powers[NUM_ANCHORS] = {0};
float qualities[NUM_ANCHORS] = {0};

device_configuration_t DEFAULT_CONFIG = {
    false, true, true, true, false,
    SFDMode::STANDARD_SFD, Channel::CHANNEL_3, DataRate::RATE_850KBPS,
    PulseFrequency::FREQ_16MHZ, PreambleLength::LEN_256, PreambleCode::CODE_3
};

interrupt_configuration_t DEFAULT_INTERRUPT_CONFIG = {
  true, true, true, false, true
};

// --- ESP-NOW and FreeRTOS Setup ---
// STREAM TO: D4:D4:DA:5C:4E:08
uint8_t broadcastAddress[] = {0xD4, 0xD4, 0xDA, 0x5C, 0x4E, 0x08};
QueueHandle_t espNowQueue;
#define MAX_PAYLOAD_SIZE 250 // ESP-NOW maximum payload limit

// Task running on Core 0 to handle ESP-NOW broadcasting asynchronously
void espNowTask(void *pvParameters) {
  char buffer[MAX_PAYLOAD_SIZE];
  for(;;) {
    // Block until a message is available in the queue
    if (xQueueReceive(espNowQueue, buffer, portMAX_DELAY) == pdTRUE) {
      esp_now_send(broadcastAddress, (uint8_t *)buffer, strlen(buffer));
    }
  }
}
// ----------------------------------

void printDistances() {
  char payload[MAX_PAYLOAD_SIZE];
  int len = 0;
  
  // Format the CSV string into the payload buffer
  for (int i = 0; i < NUM_ANCHORS; i++) {
    int added = snprintf(payload + len, sizeof(payload) - len, "%.2f,%.2f,%.2f,%.2f", 
                         distances[i], rx_powers[i], fp_powers[i], qualities[i]);
    if (added > 0 && added < sizeof(payload) - len) {
        len += added;
    } else {
        break; // Stop to prevent buffer overflow
    }

    if (i < NUM_ANCHORS - 1) {
        added = snprintf(payload + len, sizeof(payload) - len, ",");
        if (added > 0 && added < sizeof(payload) - len) {
            len += added;
        } else {
            break;
        }
    }
  }

  // Output to Serial monitor
  Serial.println(payload);

  // Send the formatted string to Core 0 task via Queue without blocking Core 1
  if (espNowQueue != NULL) {
    xQueueSend(espNowQueue, payload, 0); 
  }
}

void gotoIdle() {
  current_anchor++;
  if (current_anchor > NUM_ANCHORS) {
    printDistances();
    current_anchor = 1;
  }
  DW1000Ng::forceTRxOff();
  currentState = IDLE;
  idleTimer = millis();
}

void noteActivity() {
  lastActivity = millis();
}

void handleSent() { sentAck = true; }
void handleReceived() { receivedAck = true; }

void transmitPoll() {
  data[0] = POLL;
  data[1] = current_anchor;
  DW1000Ng::setTransmitData(data, LEN_DATA);
  DW1000Ng::startTransmit();
}

void transmitRange() {
  data[0] = RANGE;
  data[1] = current_anchor;

  byte futureTimeBytes[LENGTH_TIMESTAMP];
  timeRangeSent = DW1000Ng::getSystemTimestamp();
  timeRangeSent += DW1000NgTime::microsecondsToUWBTime(replyDelayTimeUS);
  DW1000NgUtils::writeValueToBytes(futureTimeBytes, timeRangeSent, LENGTH_TIMESTAMP);
  DW1000Ng::setDelayedTRX(futureTimeBytes);
  timeRangeSent += DW1000Ng::getTxAntennaDelay();

  DW1000NgUtils::writeValueToBytes(data + 2,  timePollSent,        LENGTH_TIMESTAMP);
  DW1000NgUtils::writeValueToBytes(data + 7,  timePollAckReceived, LENGTH_TIMESTAMP);
  DW1000NgUtils::writeValueToBytes(data + 12, timeRangeSent,       LENGTH_TIMESTAMP);

  DW1000Ng::setTransmitData(data, LEN_DATA);
  DW1000Ng::startTransmit(TransmitMode::DELAYED);
}

void setup() {
  Serial.begin(115200);

  // --- Initialize WiFi and ESP-NOW ---
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  // Create the Queue (Capacity: 10 messages, Max size: 250 bytes each)
  espNowQueue = xQueueCreate(10, MAX_PAYLOAD_SIZE);

  // Create task pinned to Core 0 (PRO_CPU) to handle wireless transmission
  xTaskCreatePinnedToCore(
    espNowTask,     // Task function
    "espNowTask",   // Name of task
    4096,           // Stack size of task
    NULL,           // Parameter of the task
    1,              // Priority of the task
    NULL,           // Task handle
    0               // Core assignment (0)
  );
  // -----------------------------------

  // DW1000 Initialization
  DW1000Ng::initialize(PIN_SS, PIN_IRQ, PIN_RST);
  DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
  DW1000Ng::applyInterruptConfiguration(DEFAULT_INTERRUPT_CONFIG);
  DW1000Ng::setNetworkId(10);
  DW1000Ng::setAntennaDelay(16440);
  DW1000Ng::attachSentHandler(handleSent);
  DW1000Ng::attachReceivedHandler(handleReceived);
  currentState = IDLE;
  idleTimer = millis();
}

void loop() {
  // The default loop() inherently runs on Core 1 (APP_CPU)
  if (currentState == IDLE) {
    if (millis() - idleTimer >= IDLE_DELAY_MS) {
      currentState = POLLING;
      expectedMsgId = POLL_ACK;
      transmitPoll();
      noteActivity();
    }
    return;
  }

  if (!sentAck && !receivedAck) {
    if (millis() - lastActivity > resetPeriod) {
      distances[current_anchor - 1] = -1.0;
      gotoIdle();
    }
    return;
  }

  if (sentAck) {
    sentAck = false;
    if (expectedMsgId == POLL_ACK) {
      timePollSent = DW1000Ng::getTransmitTimestamp();
    }
    DW1000Ng::startReceive();
  }

  if (receivedAck) {
    receivedAck = false;
    DW1000Ng::getReceivedData(data, LEN_DATA);
    byte msgId   = data[0];
    byte senderId = data[1];

    if (senderId != current_anchor || msgId != expectedMsgId) {
      DW1000Ng::startReceive();
      return;
    }

    if (msgId == POLL_ACK) {
      timePollAckReceived = DW1000Ng::getReceiveTimestamp();
      expectedMsgId = RANGE_REPORT;
      transmitRange();
      noteActivity();
    }
    else if (msgId == RANGE_REPORT) {
      float curRange;
      memcpy(&curRange, data + 2, 4);
      distances[current_anchor - 1] = curRange;
      memcpy(&rx_powers[current_anchor - 1], data + 6,  4);
      memcpy(&fp_powers[current_anchor - 1], data + 10, 4);
      memcpy(&qualities[current_anchor - 1], data + 14, 4);
      gotoIdle();
    }
    else if (msgId == RANGE_FAILED) {
      distances[current_anchor - 1] = -1.0;
      gotoIdle();
    }
  }
}