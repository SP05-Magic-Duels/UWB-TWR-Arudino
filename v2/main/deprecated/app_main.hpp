#pragma once

#include <Arduino.h>
#include <SPI.h>
#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgRanging.hpp>
#include <DW1000NgRTLS.hpp> 
#include <DW1000NgRTLSTagCentric.hpp>

#include "utils/utils.hpp"

#define NUM_ANCHORS 3
#define NUM_TAGS 2

// Anchor or tag details
class Device {

public:
    enum : uint16_t {
        ANCHOR_A = 1,
        ANCHOR_B = 2,
        ANCHOR_C = 3,
        ANCHOR_D = 4,
        TAG_A = 5,
        TAG_B = 6
    };

    uint16_t deviceTypeID;
    uint16_t shortAddress;
    char EUI[24]; // Extended Unique Identifier register. 64-bit device identifier. Register file: 0x01
    uint16_t networkId;

    // Antenna delay if additional calibration is needed
    uint16_t antenna_delay;

    Device(uint16_t deviceTypeID) {
        this->deviceTypeID = deviceTypeID;
        switch (deviceTypeID) {
            case ANCHOR_A:
                shortAddress = 1;
                strcpy(EUI, "AA:BB:CC:DD:EE:FF:00:00");
                networkId = RTLS_APP_ID;
                antenna_delay = 0;
                break;
            case ANCHOR_B:
                shortAddress = 2;
                strcpy(EUI, "AA:BB:CC:DD:EE:FF:00:01");
                networkId = RTLS_APP_ID;
                antenna_delay = 0;
                break;
            case ANCHOR_C:
                shortAddress = 3;
                strcpy(EUI, "AA:BB:CC:DD:EE:FF:00:02");
                networkId = RTLS_APP_ID;
                antenna_delay = 0;
                break;
            case ANCHOR_D:
                shortAddress = 4;
                strcpy(EUI, "AA:BB:CC:DD:EE:FF:00:03");
                networkId = RTLS_APP_ID;
                antenna_delay = 0;
                break;
            case TAG_A:
                shortAddress = 5;
                strcpy(EUI, "AA:BB:CC:DD:EE:FF:00:04");
                networkId = RTLS_APP_ID;
                antenna_delay = 0;
                break;
            case TAG_B:
                shortAddress = 6;
                strcpy(EUI, "AA:BB:CC:DD:EE:FF:00:05");
                networkId = RTLS_APP_ID;
                antenna_delay = 0;
                break;
            default:
                // Handle unknown device type
                break;
        }
    }
};

// #define IS_ANCHOR // Comment out for Tag
static const Device ANCHOR(Device::ANCHOR_A);
static const Device TAG(Device::TAG_A);

// Manual init of the SPIClass
static SPIClass vspi(VSPI);

volatile uint32_t blink_rate = 200;

void app_setup() {
    Serial.begin(115200);
    delay(1000);
    
    // Init with no interruptvspi.begin(SCK, MISO, MOSI, SS);
    vspi.begin(SPI_SCK, SPI_MISO, SPI_MOSI, SPI_CS);
    DW1000Ng::initialize(PIN_SS, 0xFF, PIN_RST, vspi);

    DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
    DW1000Ng::applySleepConfiguration(SLEEP_CONFIG);
    // DW1000Ng::setAntennaDelay(16436);

    #if defined(IS_ANCHOR)
        // DW1000Ng::enableFrameFiltering(ANCHOR_FRAME_FILTER_CONFIG);
        DW1000Ng::setDeviceAddress(ANCHOR.shortAddress);
        DW1000Ng::setEUI(ANCHOR.EUI);
        DW1000Ng::setNetworkId(ANCHOR.networkId);

        DW1000Ng::setAntennaDelay(0);
        // Setup _expectedMsgId type
        _expectedMsgId = POLL_ACK;
    #else
        // DW1000Ng::enableFrameFiltering(TAG_FRAME_FILTER_CONFIG);
        DW1000Ng::setDeviceAddress(TAG.shortAddress);
        DW1000Ng::setEUI(TAG.EUI);
        DW1000Ng::setNetworkId(TAG.networkId);

        // Setup _expectedMsgId type
        _expectedMsgId = POLL;
    #endif

    // High timeouts to ensure the Tag-Centric handshake completes
    DW1000Ng::setPreambleDetectionTimeout(64);
    DW1000Ng::setSfdDetectionTimeout(273);
    DW1000Ng::setReceiveFrameWaitTimeoutPeriod(1500); 

    Serial.println(F("Setup complete! Starting loop..."));
}

void app_loop() {

    // Calibration (set one device to 0 then divide by 2 for symmetric)
    if (!_calibrationComplete) {
        _calibrate();
        DW1000Ng::setAndSaveAntennaDelay(ANCHOR.antenna_delay);
        _calibrationComplete = true;
    }

    #if defined(IS_ANCHOR)
        // Anchor loop
        // RangeAcceptResult_TC res = DW1000NgRTLS_TC::anchorRangeAccept(NextActivity::RANGING_CONFIRM, 0xFFFF);
        RangeAcceptResult_TC res = DW1000NgRTLS_TC::anchorRangeAccept(NextActivity::ACTIVITY_FINISHED, blink_rate);
        
        if(res.success) {
            Serial.print(F("Ranging Successful with Tag"));
            Serial.println(res.source_address, HEX);
            Serial.print(F("Distance: ")); Serial.print(res.range); Serial.println(F(" m"));
            
            // Optional: Send report to a PC/Central Hub if this is a "Master" anchor
            // _transmitRangeReport(res.range, res.source_address);
        }
        Serial.println(res.success ? F("Anchor ranging successful") : F("Anchor ranging failed or no tag in range"));
        yield();

    #else
        // // Tag loop
        // RangeInfrastructureResult_TC res = DW1000NgRTLS_TC::tagTwrLocalize(1500);
        
        // if(res.success) {
        //     Serial.println(F("--- New Range Report Received ---"));
            
        //     for(int i = 0; i < 4; i++) {
        //         // Only print slots that have a valid anchor ID
        //         if(res.anchor_ids[i] != 0) {
        //             Serial.print(F("Anchor [0x"));
        //             Serial.print(res.anchor_ids[i], HEX);
        //             Serial.print(F("]: "));
        //             Serial.print(res.ranges[i]);
        //             Serial.println(F(" m"));
        //         }
        //     }

        //     /* * TRILATERATION: 
        //      * You can now call your math function right here on the tag!
        //      * double x, y;
        //      * calculatePosition(res.ranges, x, y); 
        //      */

        //     if(res.new_blink_rate > 0) blink_rate = res.new_blink_rate;
        // } else {
        //     // Serial.println(F("Ranging failed or no anchors in range."));
        // }
        // Serial.println(res.success ? F("Tag ranging successful") : F("Tag ranging failed or no anchors in range"));

        // delay(blink_rate); // Wait for next blink cycle
        // yield();

    #endif
}