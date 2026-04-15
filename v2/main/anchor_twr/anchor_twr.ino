#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgRanging.hpp>

// --- MULTI-ANCHOR CONFIGURATION ---
#define ANCHOR_NUM 1 // Change this for each anchor!
// ----------------------------------

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

uint64_t timePollSent;
uint64_t timePollReceived;
uint64_t timePollAckSent;
uint64_t timePollAckReceived;
uint64_t timeRangeSent;
uint64_t timeRangeReceived;

#define LEN_DATA 20 
byte data[LEN_DATA];

uint32_t lastActivity;
uint32_t resetPeriod = 250; 
uint16_t replyDelayTimeUS = 2500; //7000; 

device_configuration_t DEFAULT_CONFIG = {
    false, true, true, true, false,
    SFDMode::STANDARD_SFD, Channel::CHANNEL_3, DataRate::RATE_850KBPS,
    PulseFrequency::FREQ_16MHZ, PreambleLength::LEN_256, PreambleCode::CODE_3
};

interrupt_configuration_t DEFAULT_INTERRUPT_CONFIG = {
    true, true, true, false, true
};

void receiver() {
    DW1000Ng::forceTRxOff();
    DW1000Ng::startReceive();
}

void noteActivity() {
    lastActivity = millis();
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

void transmitRangeReport(float curRange, uint8_t txPower) {
    data[0] = RANGE_REPORT;
    data[1] = ANCHOR_NUM;
    memcpy(data + 2, &curRange, 4); // bytes 2-5: distance (float)
    data[6] = txPower;              // byte 6:    TX power register value
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
    delay(1000);
    Serial.print(F("### DW1000Ng Anchor #"));
    Serial.print(ANCHOR_NUM);
    Serial.println(F(" ###"));
    
    DW1000Ng::initialize(PIN_SS, PIN_IRQ, PIN_RST);
    DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
    DW1000Ng::applyInterruptConfiguration(DEFAULT_INTERRUPT_CONFIG);
    DW1000Ng::setDeviceAddress(ANCHOR_NUM);
    DW1000Ng::setAntennaDelay(16436); // 16436
    
    DW1000Ng::attachSentHandler(handleSent);
    DW1000Ng::attachReceivedHandler(handleReceived);
    
    receiver();
    noteActivity();
}

void loop() {
    int32_t curMillis = millis();
    if (!sentAck && !receivedAck) {
        if (curMillis - lastActivity > resetPeriod) {
            resetInactive();
        }
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
        
        // Soft-ignore packets not meant for this anchor. Do NOT forceTRxOff here.
        if (targetId != ANCHOR_NUM) {
            DW1000Ng::startReceive();
            return;
        }

        if (msgId != expectedMsgId) protocolFailed = true;
        
        if (msgId == POLL) {
            protocolFailed = false;
            timePollReceived = DW1000Ng::getReceiveTimestamp();
            expectedMsgId = RANGE;
            transmitPollAck();
            noteActivity();
        }
        else if (msgId == RANGE) {
            timeRangeReceived = DW1000Ng::getReceiveTimestamp();
            expectedMsgId = POLL;
            if (!protocolFailed) {
                timePollSent = DW1000NgUtils::bytesAsValue(data + 2, LENGTH_TIMESTAMP);
                timePollAckReceived = DW1000NgUtils::bytesAsValue(data + 7, LENGTH_TIMESTAMP);
                timeRangeSent = DW1000NgUtils::bytesAsValue(data + 12, LENGTH_TIMESTAMP);
                
                double distance = DW1000NgRanging::computeRangeAsymmetric(timePollSent,
                                                            timePollReceived, 
                                                            timePollAckSent, 
                                                            timePollAckReceived, 
                                                            timeRangeSent, 
                                                            timeRangeReceived);
                //distance = DW1000NgRanging::correctRange(distance); // Don't apply bias?

                // Read the TX power control register so the tag can log it
                // alongside the distance. getTxPower() returns the current
                // value of the TX_POWER register (e.g. 0x1F for max power).
                uint8_t txPower = DW1000Ng::getReceivePower();
                transmitRangeReport(distance * DISTANCE_OF_RADIO_INV, txPower);
            } else {
                transmitRangeFailed();
            }
            noteActivity();
        }
    }
}