#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgTime.hpp>
#include <DW1000NgConstants.hpp>

// --- MULTI-ANCHOR CONFIGURATION ---
#define NUM_ANCHORS 4
// ----------------------------------

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
uint32_t resetPeriod = 30; // Quick timeout for missed packets
uint16_t replyDelayTimeUS = 7000; 

// --- TDMA State Machine Variables ---
enum TagState { IDLE, POLLING };
TagState currentState = IDLE;
uint32_t idleTimer = 0;
const uint32_t IDLE_DELAY_MS = 10; // 10ms radio silence between polling anchors

uint8_t current_anchor = 1;
float   distances[NUM_ANCHORS] = {0};
uint8_t txPowers[NUM_ANCHORS]  = {0}; // TX power register value reported by each anchor

device_configuration_t DEFAULT_CONFIG = {
    false, true, true, true, false,
    SFDMode::STANDARD_SFD, Channel::CHANNEL_3, DataRate::RATE_850KBPS,
    PulseFrequency::FREQ_16MHZ, PreambleLength::LEN_256, PreambleCode::CODE_3
};

interrupt_configuration_t DEFAULT_INTERRUPT_CONFIG = {
    true, true, true, false, true
};

void noteActivity() {
    lastActivity = millis();
}

void printDistances() {
    for (int i = 0; i < NUM_ANCHORS; i++) {
        Serial.print(distances[i]);
        Serial.print("m@");
        Serial.print(txPowers[i]);
        if (i < NUM_ANCHORS - 1) Serial.print(",");
    }
    Serial.println();
}

// Moves to the next anchor and enforces a cool-down period
void gotoIdle() {
    current_anchor++;
    if (current_anchor > NUM_ANCHORS) {
        printDistances();
        current_anchor = 1; 
    }
    DW1000Ng::forceTRxOff(); // Shut down the radio to clear the airwaves
    currentState = IDLE;
    idleTimer = millis();
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

    DW1000NgUtils::writeValueToBytes(data + 2, timePollSent, LENGTH_TIMESTAMP);
    DW1000NgUtils::writeValueToBytes(data + 7, timePollAckReceived, LENGTH_TIMESTAMP);
    DW1000NgUtils::writeValueToBytes(data + 12, timeRangeSent, LENGTH_TIMESTAMP);
    
    DW1000Ng::setTransmitData(data, LEN_DATA);
    DW1000Ng::startTransmit(TransmitMode::DELAYED);
}

void app_setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println(F("### DW1000Ng Tag - TDMA Multi-Anchor ###"));
    
    DW1000Ng::initialize(PIN_SS, PIN_IRQ, PIN_RST);
    DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
    DW1000Ng::applyInterruptConfiguration(DEFAULT_INTERRUPT_CONFIG);
    DW1000Ng::setNetworkId(10);
    DW1000Ng::setAntennaDelay(16436);
    
    DW1000Ng::attachSentHandler(handleSent);
    DW1000Ng::attachReceivedHandler(handleReceived);
    
    // Start in IDLE state to kick off the cycle
    currentState = IDLE;
    idleTimer = millis();
}

void app_loop() {
    // --- TDMA Delay Logic ---
    if (currentState == IDLE) {
        if (millis() - idleTimer >= IDLE_DELAY_MS) {
            currentState = POLLING;
            expectedMsgId = POLL_ACK;
            transmitPoll();
            noteActivity();
        }
        return; 
    }

    // --- Polling Logic ---
    if (!sentAck && !receivedAck) {
        if (millis() - lastActivity > resetPeriod) {
            distances[current_anchor - 1] = -1.0; 
            gotoIdle(); 
        }
        return;
    }
    
    if (sentAck) {
        sentAck = false;
        DW1000Ng::startReceive();
    }
    
    if (receivedAck) {
        receivedAck = false;
        DW1000Ng::getReceivedData(data, LEN_DATA);
        
        byte msgId = data[0];
        byte senderId = data[1];

        // Soft ignore for stray packets
        if (senderId != current_anchor || msgId != expectedMsgId) {
            DW1000Ng::startReceive(); 
            return;
        }
        
        if (msgId == POLL_ACK) {
            timePollSent = DW1000Ng::getTransmitTimestamp();
            timePollAckReceived = DW1000Ng::getReceiveTimestamp();
            expectedMsgId = RANGE_REPORT;
            transmitRange();
            noteActivity();
        } 
        else if (msgId == RANGE_REPORT) {
            float curRange;
            memcpy(&curRange, data + 2, 4);                    // bytes 2-5: distance
            distances[current_anchor - 1] = curRange / DISTANCE_OF_RADIO_INV;
            txPowers[current_anchor - 1]  = data[6];           // byte 6: anchor TX power
            gotoIdle(); // Success! Wait, then move to next anchor
        } 
        else if (msgId == RANGE_FAILED) {
            distances[current_anchor - 1] = -1.0; 
            gotoIdle();
        }
    }
}