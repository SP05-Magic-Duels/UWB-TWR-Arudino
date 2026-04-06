/*
 * TagCentricRTLS_Tag_TWR.ino
 *
 * TAG sketch — tag-centric TWR RTLS.
 *
 * The tag owns the full ranging cycle:
 *   1. Poll each anchor in `ANCHOR_ADDRESSES` sequentially.
 *   2. Complete the asymmetric TWR exchange with each anchor.
 *   3. Receive the computed range back from each anchor in the control frame.
 *   4. Run trilateration locally and print the (x, y) position.
 *
 * Anchor sketches must run TagCentricRTLS_Anchor_TWR.ino and be configured
 * with matching addresses and the same RF parameters.
 *
 * Wire format note
 * ─────────────────
 * For RANGING_CONFIRM frames  (non-last anchors): bytes 11–12 carry the
 *   range in millimetres (uint16_t, little-endian) instead of the next-anchor
 *   address.  The tag uses its own `ANCHOR_ADDRESSES` list to know which
 *   anchor to poll next.
 * For ACTIVITY_FINISHED frames (last anchor):    bytes 11–12 carry the range
 *   in millimetres.  There is no blink-rate advisory in this implementation
 *   (new_blink_rate stays 0); add it to bytes 13–14 if you need it.
 */

#include <Arduino.h>
#include <SPI.h>

#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgTime.hpp>
#include <DW1000NgConstants.hpp>
#include <DW1000NgRanging.hpp>
#include <DW1000NgRTLS.hpp>
#include "DW1000NgRTLSTC.hpp"

#include "utils/configs.hpp"

// Device identity
const char EUI[] = "AA:BB:CC:DD:EE:FF:00:04"; // must be unique per tag

// Anchor list (short addresses, order determines polling sequence)
// Anchor A = 0x0001, Anchor B = 0x0002, Anchor C = 0x0003
// The last entry will receive NextActivity::ACTIVITY_FINISHED;
// all others receive NextActivity::RANGING_CONFIRM.
uint16_t ANCHOR_ADDRESSES[] = {0x0001, 0x0002, 0x0003};
const uint8_t NUM_ANCHORS   = sizeof(ANCHOR_ADDRESSES) / sizeof(ANCHOR_ADDRESSES[0]);

// Known anchor positions (metres) — must match your physical layout
typedef struct { double x; double y; } Position;

const Position ANCHOR_POSITIONS[3] = {
    {0.0, 0.0},   // Anchor A  (origin)
    {3.0, 0.0},   // Anchor B
    {3.0, 2.5}    // Anchor C
};

// TWR timing (µs) — 1500 µs is safe for 8–80 MHz devices
const uint16_t FINAL_MSG_DELAY_US = 1500;

// Blink / poll rate
volatile uint32_t blink_rate = 200; // ms between localization cycles

static SPIClass vspi(VSPI);

// 2D Trilateration
// Uses the three-anchor linear-algebra method from:
// https://math.stackexchange.com/questions/884807
void calculate2DPosition(double r0, double r1, double r2, double &x, double &y) {
    double ax = ANCHOR_POSITIONS[0].x, ay = ANCHOR_POSITIONS[0].y;
    double bx = ANCHOR_POSITIONS[1].x, by = ANCHOR_POSITIONS[1].y;
    double cx = ANCHOR_POSITIONS[2].x, cy = ANCHOR_POSITIONS[2].y;

    double A = (-2*ax) + (2*bx);
    double B = (-2*ay) + (2*by);
    double C = (r0*r0) - (r1*r1) - (ax*ax) + (bx*bx) - (ay*ay) + (by*by);
    double D = (-2*bx) + (2*cx);
    double E = (-2*by) + (2*cy);
    double F = (r1*r1) - (r2*r2) - (bx*bx) + (cx*cx) - (by*by) + (cy*cy);

    double denom = (E*A - B*D);
    if (abs(denom) < 1e-9) {
        // Anchors are collinear — cannot trilaterate.
        x = 0; y = 0;
        return;
    }
    x = (C*E - F*B) / denom;
    y = (C*D - A*F) / (B*D - A*E);
}

// 3D trilateration (optional; not used in this sketch)
void calculate3DPosition(double r0, double r1, double r2, double &x, double &y, double &z) {
    // Implement if needed, using the known Z coordinates of the anchors.
    x = 0; y = 0; z = 0;
}

void app_setup() {
    Serial.begin(115200);
    Serial.println(F("### Tag-Centric RTLS Tag ###"));

    // Init with no interrupt
    vspi.begin(SPI_SCK, SPI_MISO, SPI_MOSI, SPI_CS);
    DW1000Ng::initialize(PIN_SS, 0xFF, PIN_RST, vspi);

    DW1000Ng::applyConfiguration(DEFAULT_CONFIG);
    DW1000Ng::enableFrameFiltering(TAG_FRAME_FILTER_CONFIG);

    DW1000Ng::setEUI(EUI);
    DW1000Ng::setNetworkId(RTLS_APP_ID);
    DW1000Ng::setAntennaDelay(16436);

    DW1000Ng::applySleepConfiguration(SLEEP_CONFIG);

    // Timeouts — keep generous during bring-up; tighten after validation.
    DW1000Ng::setPreambleDetectionTimeout(64);
    DW1000Ng::setSfdDetectionTimeout(273);
    DW1000Ng::setReceiveFrameWaitTimeoutPeriod(2000);

    Serial.println(F("Setup complete."));

    char msg[128];
    DW1000Ng::getPrintableDeviceIdentifier(msg);
    Serial.print(F("Device ID: ")); Serial.println(msg);
    DW1000Ng::getPrintableExtendedUniqueIdentifier(msg);
    Serial.print(F("Unique ID: ")); Serial.println(msg);
    DW1000Ng::getPrintableNetworkIdAndShortAddress(msg);
    Serial.print(F("Network ID & Address: ")); Serial.println(msg);
}

void app_loop() {
    // Deep sleep between cycles (saves ~18 mA)
    DW1000Ng::deepSleep();
    delay(blink_rate);
    DW1000Ng::spiWakeup();
    DW1000Ng::setEUI(EUI); // EUI must be re-applied after wake

    // Run a full localization cycle
    RangeInfrastructureResult_TC res = DW1000NgRTLS_TC::tagTwrLocalize(
        ANCHOR_ADDRESSES,
        NUM_ANCHORS,
        FINAL_MSG_DELAY_US
    );

    if (res.success && res.num_anchors >= 3) {
        Serial.println(F("--- Range Report ---"));
        for (uint8_t i = 0; i < res.num_anchors; i++) {
            Serial.print(F("Anchor 0x"));
            Serial.print(res.anchor_ids[i], HEX);
            Serial.print(F(": "));
            Serial.print(res.ranges[i], 3);
            Serial.println(F(" m"));
        }

        double x, y;
        calculate2DPosition(res.ranges[0], res.ranges[1], res.ranges[2], x, y);
        Serial.print(F("Position — x: ")); Serial.print(x, 3);
        Serial.print(F("  y: "));          Serial.println(y, 3);

        if (res.new_blink_rate > 0) {
            blink_rate = res.new_blink_rate;
        }
    } else {
        Serial.println(F("Ranging failed or insufficient anchors."));
    }
}
