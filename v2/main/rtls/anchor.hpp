/*
 * TagCentricRTLS_Anchor_TWR.ino
 *
 * ANCHOR sketch — tag-centric TWR RTLS.
 *
 * Each anchor is a passive responder.  It:
 *   1. Listens for a Poll from the tag.
 *   2. Sends a Response-to-Poll (gives the tag its timestamps).
 *   3. Receives the Final message.
 *   4. Computes the range using asymmetric TWR.
 *   5. Packs the range (in mm) into bytes 11–12 of a RANGING_CONFIRM or
 *      ACTIVITY_FINISHED frame and transmits it back to the tag.
 *   6. Returns to step 1.
 *
 * Per-anchor configuration — edit the three constants below and re-flash:
 *   THIS_ANCHOR_ID   : must match the entry in the tag's ANCHOR_ADDRESSES[].
 *   EUI              : must be unique per anchor.
 *   IS_LAST_ANCHOR   : set true on the last anchor in the tag's poll list.
 */

#include <Arduino.h>
#include <SPI.h>

#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgRanging.hpp>
#include <DW1000NgRTLS.hpp>
#include "DW1000NgRTLSTC.hpp"

#include "utils/configs.hpp"

// Per-anchor identity 
// const uint16_t THIS_ANCHOR_ID = 1;                        // 1, 2, or 3
// const char     EUI[]          = "AA:BB:CC:DD:EE:FF:00:01"; // unique per anchor
// const bool     IS_LAST_ANCHOR = false;                     // true only on last anchor

// const uint16_t THIS_ANCHOR_ID = 2;                        // 1, 2, or 3
// const char     EUI[]          = "AA:BB:CC:DD:EE:FF:00:02"; // unique per anchor
// const bool     IS_LAST_ANCHOR = false;                     // true only on last anchor

const uint16_t THIS_ANCHOR_ID = 3;                        // 1, 2, or 3
const char     EUI[]          = "AA:BB:CC:DD:EE:FF:00:03"; // unique per anchor
const bool     IS_LAST_ANCHOR = true;                      // true only on last anchor

static SPIClass vspi(VSPI);

void app_setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.print(F("### Tag-Centric RTLS Anchor 0x"));
    Serial.print(THIS_ANCHOR_ID, HEX);
    Serial.println(F(" ###"));

    // Init with no interrupt
    vspi.begin(SPI_SCK, SPI_MISO, SPI_MOSI, SPI_CS);
    DW1000Ng::initialize(PIN_SS, 0xFF, PIN_RST, vspi);

    DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
    DW1000Ng::enableFrameFiltering(ANCHOR_FRAME_FILTER_CONFIG);

    DW1000Ng::setEUI(EUI);
    DW1000Ng::setNetworkId(RTLS_APP_ID);
    DW1000Ng::setDeviceAddress(THIS_ANCHOR_ID);
    DW1000Ng::setAntennaDelay(16436); // TODO: calibrate this

    // Anchors just wait — keep timeouts generous.
    DW1000Ng::setPreambleDetectionTimeout(64);
    DW1000Ng::setSfdDetectionTimeout(273);
    DW1000Ng::setReceiveFrameWaitTimeoutPeriod(5000);

    Serial.println(F("Setup complete. Listening for tag polls..."));

    char msg[128];
    DW1000Ng::getPrintableDeviceIdentifier(msg);
    Serial.print(F("Device ID: ")); Serial.println(msg);
    DW1000Ng::getPrintableExtendedUniqueIdentifier(msg);
    Serial.print(F("Unique ID: ")); Serial.println(msg);
    DW1000Ng::getPrintableNetworkIdAndShortAddress(msg);
    Serial.print(F("Network ID & Address: ")); Serial.println(msg);
}

void app_loop() {
    // Decide what the tag should do after this exchange.
    // All anchors except the last send RANGING_CONFIRM (tag has more anchors to poll).
    // The last anchor sends ACTIVITY_FINISHED (tag's cycle is complete).
    NextActivity next = IS_LAST_ANCHOR
        ? NextActivity::ACTIVITY_FINISHED
        : NextActivity::RANGING_CONFIRM;

    // anchorRangeAccept blocks until Poll -> Response -> Final completes (or times
    // out).  It computes the range and packs it as range_mm into the control
    // frame automatically — the `value` argument is ignored by this TC version.
    RangeAcceptResult_TC res = DW1000NgRTLS_TC::anchorRangeAccept(next, 0);

    if (res.success) {
        Serial.print(F("Ranged tag 0x"));
        Serial.print(res.source_address, HEX);
        Serial.print(F(": "));
        Serial.print(res.range, 3);
        Serial.println(F(" m"));
    }
    // On failure (timeout/bad frame) we simply loop back and listen again.
}
