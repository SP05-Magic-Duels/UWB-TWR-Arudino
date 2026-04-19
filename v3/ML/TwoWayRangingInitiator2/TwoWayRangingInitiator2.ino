#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgTime.hpp>
#include <DW1000NgConstants.hpp>
#include <WiFi.h>
#include <esp_now.h>

// --- IMU ---
#include <Wire.h>
#include <FastIMU.h>

#define IMU_ADDRESS 0x69  // Adjust to 0x68 if needed
BMI160 IMU;

// Reference for IMU data structures
// struct calData {
// 	bool valid;
// 	float accelBias[3];
// 	float gyroBias[3];
// 	float magBias[3];
// 	float magScale[3];
// };
calData calib = { 
    true, 
    {1.0f, 0.0f, 1.0f}, 
    {-54.90f, -124.95f, 33.08f}, 
    {0.0f, 0.0f, 0.0f}, 
    {1.0f, 1.0f, 1.0f} 
};
AccelData accelData;
GyroData gyroData;
// -----------

#define NUM_ANCHORS 8

// Byte budget (worst case per field):
//   distances : "XX.XXXX,"  = 8 chars x 8 = 64  (%.4f — full precision needed)
//   rx_powers : "-XX.XX,"   = 7 chars x 8 = 56  (%.2f — dBm, 2dp is sufficient)
//   fp_powers : "-XX.XX,"   = 7 chars x 8 = 56  (%.2f — dBm, 2dp is sufficient)
//   qualities : "X.XX,"     = 5 chars x 8 = 40  (%.2f — ratio, 2dp is sufficient)
//   IMU x6    : "XX.X,"     = 5 chars x 6 = 30  (%.1f — passthrough only, low priority)
//   null terminator         =              =  1
//   ─────────────────────────────────────────────
//   Total worst case                       = 247 bytes ✅ under ESP-NOW 250 byte limit
#define MAX_PAYLOAD_SIZE 250

// --- UWB Variables ---
const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS  = 4;

#define POLL         0
#define POLL_ACK     1
#define RANGE        2
#define RANGE_REPORT 3
#define RANGE_FAILED 255

volatile byte    expectedMsgId = POLL_ACK;
volatile boolean sentAck       = false;
volatile boolean receivedAck   = false;

uint64_t timePollSent;
uint64_t timePollAckReceived;
uint64_t timeRangeSent;

#define LEN_DATA 20
byte data[LEN_DATA];

uint32_t lastActivity;
uint32_t resetPeriod      = 30;
uint16_t replyDelayTimeUS = 3000;

enum TagState { IDLE, POLLING };
TagState currentState = IDLE;
uint32_t idleTimer    = 0;
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

// --- ESP-NOW ---
// STREAM TO: D4:D4:DA:5C:4E:08
uint8_t broadcastAddress[] = {0xD4, 0xD4, 0xDA, 0x5C, 0x4E, 0x08};
QueueHandle_t espNowQueue;

void espNowTask(void *pvParameters) {
  char txData[MAX_PAYLOAD_SIZE];
  for (;;) {
    if (xQueueReceive(espNowQueue, &txData, portMAX_DELAY) == pdTRUE) {
      esp_now_send(broadcastAddress, (uint8_t *)txData, strlen(txData));
    }
  }
}

void printDistances() {
  // 1. Read IMU
  IMU.update();
  IMU.getAccel(&accelData);
  IMU.getGyro(&gyroData);

  char payload[MAX_PAYLOAD_SIZE];
  int len   = 0;
  int added = 0;

  // 2. Interleaved UWB per anchor:
  //      dist   = %.4f  (full precision, critical for trilateration)
  //      rx/fp  = %.2f  (dBm signal powers, 2dp is sufficient)
  //      qual   = %.2f  (quality ratio, 2dp is sufficient)
  for (int i = 0; i < NUM_ANCHORS; i++) {
    added = snprintf(payload + len, sizeof(payload) - len,
                     "%.4f,%.2f,%.2f,%.2f,",
                     distances[i], rx_powers[i], fp_powers[i], qualities[i]);
    if (added > 0 && added < (int)(sizeof(payload) - len)) {
      len += added;
    } else { break; }
  }

  // 3. Append IMU at end: gyroX,gyroY,gyroZ,accelX,accelY,accelZ
  //    %.1f — IMU is passthrough only for spike/fall detection, low precision needed
  //    No trailing comma on the last field
  added = snprintf(payload + len, sizeof(payload) - len,
                   "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f",
                   gyroData.gyroX,   gyroData.gyroY,   gyroData.gyroZ,
                   accelData.accelX, accelData.accelY, accelData.accelZ);
  if (added > 0 && added < (int)(sizeof(payload) - len)) {
    len += added;
  }

  // 4. Output to Serial
  Serial.println(payload);

  // 5. Send to Core 0 ESP-NOW task via queue
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
  idleTimer    = millis();
}

void noteActivity() {
  lastActivity = millis();
}

void handleSent()     { sentAck     = true; }
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
  timeRangeSent  = DW1000Ng::getSystemTimestamp();
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

  // --- IMU ---
  Wire.begin();
  int err = IMU.init(calib, IMU_ADDRESS);
  if (err != 0) {
    Serial.print("Error initializing BMI160! Code: ");
    Serial.println(err);
  } else {
    Serial.println("BMI160 initialized successfully!");
  }
  // Serial.println("Calibrating IMU... Do not move it.");
  delay(300); // Allow some time for the IMU to stabilize after initialization

  // IMU.calibrateAccelGyro(&calib);
  // Serial.print("Calibration data - Accel Bias: [");
  // Serial.print(calib.accelBias[0], 2); Serial.print(", ");
  // Serial.print(calib.accelBias[1], 2); Serial.print(", ");
  // Serial.print(calib.accelBias[2], 2); Serial.println("]");
  // Serial.print("Calibration data - Gyro Bias: [");
  // Serial.print(calib.gyroBias[0], 2); Serial.print(", ");
  // Serial.print(calib.gyroBias[1], 2); Serial.print(", ");
  // Serial.print(calib.gyroBias[2], 2); Serial.println("]");
  // Serial.println("IMU calibration complete.");

  // --- WiFi + ESP-NOW ---
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

  // Queue item size matches char[MAX_PAYLOAD_SIZE]
  espNowQueue = xQueueCreate(10, MAX_PAYLOAD_SIZE);

  xTaskCreatePinnedToCore(
    espNowTask,   // Task function
    "espNowTask", // Name
    4096,         // Stack size
    NULL,         // Parameters
    1,            // Priority
    NULL,         // Handle
    0             // Core 0
  );

  // --- DW1000 ---
  DW1000Ng::initialize(PIN_SS, PIN_IRQ, PIN_RST);
  DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
  DW1000Ng::applyInterruptConfiguration(DEFAULT_INTERRUPT_CONFIG);
  DW1000Ng::setNetworkId(10);
  DW1000Ng::setAntennaDelay(16440);
  DW1000Ng::attachSentHandler(handleSent);
  DW1000Ng::attachReceivedHandler(handleReceived);
  currentState = IDLE;
  idleTimer    = millis();
}

void loop() {
  // Runs on Core 1 (APP_CPU)
  if (currentState == IDLE) {
    if (millis() - idleTimer >= IDLE_DELAY_MS) {
      currentState  = POLLING;
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
    byte msgId    = data[0];
    byte senderId = data[1];

    if (senderId != current_anchor || msgId != expectedMsgId) {
      DW1000Ng::startReceive();
      return;
    }

    if (msgId == POLL_ACK) {
      timePollAckReceived = DW1000Ng::getReceiveTimestamp();
      expectedMsgId       = RANGE_REPORT;
      transmitRange();
      noteActivity();
    }
    else if (msgId == RANGE_REPORT) {
      float curRange;
      memcpy(&curRange,                      data + 2,  4);
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
