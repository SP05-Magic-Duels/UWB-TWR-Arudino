#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgRanging.hpp>
#include <math.h>

#define ANCHOR_NUM 4 // Set to 1, 2, 3, or 4 for each physical anchor

const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS = 4;

#define POLL 0
#define POLL_ACK 1
#define RANGE 2
#define RANGE_REPORT 3
#define RANGE_FAILED 255

volatile byte expectedMsgId = POLL;
volatile boolean sentAck = false;
volatile boolean receivedAck = false;
boolean protocolFailed = false;

uint64_t timePollSent, timePollReceived;
uint64_t timePollAckSent, timePollAckReceived;
uint64_t timeRangeSent, timeRangeReceived;

#define LEN_DATA 20
byte data[LEN_DATA];
uint32_t lastActivity;
uint32_t resetPeriod = 250;
uint16_t replyDelayTimeUS = 3000;

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

void noteActivity() { lastActivity = millis(); }

void receiver() {
  DW1000Ng::forceTRxOff();
  DW1000Ng::startReceive();
}

void resetInactive() {
  expectedMsgId = POLL;
  receiver();
  noteActivity();
}

void handleSent() { sentAck = true; }
void handleReceived() { receivedAck = true; }

void transmitPollAck() {
  data[0] = POLL_ACK;
  data[1] = ANCHOR_NUM;
  DW1000Ng::setTransmitData(data, LEN_DATA);
  DW1000Ng::startTransmit();
}

void transmitRangeReport(float curRange, float rxPower, float fpPower, float quality) {
  data[0] = RANGE_REPORT;
  data[1] = ANCHOR_NUM;
  memcpy(data + 2, &curRange, 4);
  memcpy(data + 6, &rxPower, 4);
  memcpy(data + 10, &fpPower, 4);
  memcpy(data + 14, &quality, 4);
  DW1000Ng::setTransmitData(data, LEN_DATA);
  DW1000Ng::startTransmit();
}

void transmitRangeFailed() {
  data[0] = RANGE_FAILED;
  data[1] = ANCHOR_NUM;
  DW1000Ng::setTransmitData(data, LEN_DATA);
  DW1000Ng::startTransmit();
}

void setup() {
  Serial.begin(115200);
  DW1000Ng::initialize(PIN_SS, PIN_IRQ, PIN_RST);
  DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
  DW1000Ng::applyInterruptConfiguration(DEFAULT_INTERRUPT_CONFIG);
  DW1000Ng::setDeviceAddress(ANCHOR_NUM);
  DW1000Ng::setAntennaDelay(16440);
  DW1000Ng::attachSentHandler(handleSent);
  DW1000Ng::attachReceivedHandler(handleReceived);
  receiver();
  noteActivity();
}

void loop() {
  uint32_t curMillis = millis();
  if (!sentAck && !receivedAck) {
    if (curMillis - lastActivity > resetPeriod) resetInactive();
    return;
  }

  if (sentAck) {
    sentAck = false;
    if (data[0] == POLL_ACK) {
      timePollAckSent = DW1000Ng::getTransmitTimestamp();
      noteActivity();
    }
    DW1000Ng::startReceive();
  }

  if (receivedAck) {
    receivedAck = false;
    DW1000Ng::getReceivedData(data, LEN_DATA);
    byte msgId = data[0];
    byte targetId = data[1];

    // Check if this packet is addressed to us
    if (targetId != ANCHOR_NUM) {
      DW1000Ng::startReceive();
      return;
    }

    if (msgId == POLL) {
      protocolFailed = false;
      timePollReceived = DW1000Ng::getReceiveTimestamp();
      expectedMsgId = RANGE;
      transmitPollAck();
      noteActivity();

    } else if (msgId == RANGE && expectedMsgId == RANGE) {
      // FIX: Check expectedMsgId BEFORE overwriting it, and only proceed if matched.
      // Previously, expectedMsgId was set to POLL first, making the check always fail
      // and causing protocolFailed to never be cleared — always sending RANGE_FAILED.
      timeRangeReceived = DW1000Ng::getReceiveTimestamp();
      expectedMsgId = POLL;
      protocolFailed = false;

      // FIX: Timestamps in the RANGE packet start at data+2, not data+1.
      // data[0]=msgId, data[1]=anchorId are the 2-byte header.
      // Previously offsets +1/+6/+11 caused data[1] (anchorId) to be read as
      // part of timePollSent, corrupting all three timestamps and causing
      // the DW1000 checksum/stack crash on the responder.
      timePollSent        = DW1000NgUtils::bytesAsValue(data + 2,  LENGTH_TIMESTAMP);
      timePollAckReceived = DW1000NgUtils::bytesAsValue(data + 7,  LENGTH_TIMESTAMP);
      timeRangeSent       = DW1000NgUtils::bytesAsValue(data + 12, LENGTH_TIMESTAMP);

      double raw_dist = DW1000NgRanging::computeRangeAsymmetric(
        timePollSent, timePollReceived,
        timePollAckSent, timePollAckReceived,
        timeRangeSent, timeRangeReceived
      );
      raw_dist *= 0.87; // Bias correction

      float rx_p  = DW1000Ng::getReceivePower();
      float fp_p  = DW1000Ng::getFirstPathPower();
      float qual  = DW1000Ng::getReceiveQuality();

      // FIX: Send raw_dist directly (in metres). The initiator will receive it as-is.
      // Previously raw_dist was multiplied by DISTANCE_OF_RADIO_INV here AND then
      // divided by it on the initiator side, causing the distance to be scaled twice.
      transmitRangeReport((float)raw_dist, rx_p, fp_p, qual);
      noteActivity();

    } else {
      // Unexpected message — reset and listen again
      protocolFailed = true;
      expectedMsgId = POLL;
      DW1000Ng::startReceive();
    }
  }
}
