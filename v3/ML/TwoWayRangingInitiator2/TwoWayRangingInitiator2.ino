#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgTime.hpp>
#include <DW1000NgConstants.hpp>

#define NUM_ANCHORS 4

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

// device_configuration_t DEFAULT_CONFIG = {
//   false, true, true, true, false,
//   SFDMode::STANDARD_SFD, Channel::CHANNEL_5, DataRate::RATE_6800KBPS,
//   PulseFrequency::FREQ_64MHZ, PreambleLength::LEN_128, PreambleCode::CODE_9
// };

device_configuration_t DEFAULT_CONFIG = {
    false, true, true, true, false,
    SFDMode::STANDARD_SFD, Channel::CHANNEL_3, DataRate::RATE_850KBPS,
    PulseFrequency::FREQ_16MHZ, PreambleLength::LEN_256, PreambleCode::CODE_3
};

interrupt_configuration_t DEFAULT_INTERRUPT_CONFIG = {
  true, true, true, false, true
};

void printDistances() {
  for (int i = 0; i < NUM_ANCHORS; i++) {
    Serial.print(distances[i]);
    Serial.print(",");
    Serial.print(rx_powers[i]);
    Serial.print(",");
    Serial.print(fp_powers[i]);
    Serial.print(",");
    Serial.print(qualities[i]);
    if (i < NUM_ANCHORS - 1) Serial.print(",");
  }
  Serial.println();
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

  // FIX: Timestamps now packed starting at data+2 (after the 2-byte header).
  // Previously data+1 overlapped with data[1] (the anchor ID byte), causing the
  // responder to read a corrupted anchorId and garbage timestamps, which triggered
  // the DW1000 internal checksum failure / core dump on the responder side.
  DW1000NgUtils::writeValueToBytes(data + 2,  timePollSent,        LENGTH_TIMESTAMP);
  DW1000NgUtils::writeValueToBytes(data + 7,  timePollAckReceived, LENGTH_TIMESTAMP);
  DW1000NgUtils::writeValueToBytes(data + 12, timeRangeSent,       LENGTH_TIMESTAMP);

  DW1000Ng::setTransmitData(data, LEN_DATA);
  DW1000Ng::startTransmit(TransmitMode::DELAYED);
}

void setup() {
  Serial.begin(115200);
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
    // Capture POLL transmit timestamp immediately after the POLL is sent,
    // before any new TX (transmitRange) can overwrite the hardware register.
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
      // FIX: Receive raw distance directly (metres); no DISTANCE_OF_RADIO_INV
      // division needed here. Previously the responder multiplied by the factor
      // and the initiator divided by it, double-applying the conversion.
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
