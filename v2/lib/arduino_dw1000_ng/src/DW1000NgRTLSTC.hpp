/*
 * MIT License
 * 
 * Copyright (c) 2018 Michele Biondi, Andrea Salvatori
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

/*
 * DW1000NgRTLS_TC.hpp
 * Tag-Centric TWR RTLS Extension
 *
 * In this architecture the TAG owns the full ranging cycle:
 *   1. Tag polls each anchor directly (no Blink / RangingInitiation phase).
 *   2. Each anchor is a passive responder — it timestamps and replies, nothing more.
 *   3. The tag collects all ranges and runs trilateration itself.
 *
 */

#pragma once

#include <Arduino.h>
#include "DW1000NgRTLS.hpp"   // reuses existing frame constants and helpers

// Maximum anchors the tag will range in one localization cycle.
#ifndef TC_MAX_ANCHORS
#define TC_MAX_ANCHORS 4
#endif

/** Returned by anchorRangeAccept_TC — the anchor just reports success/range. */
typedef struct RangeAcceptResult_TC {
    boolean  success;
    double   range;
    uint16_t source_address;                // short address of the tag that polled us
} RangeAcceptResult_TC;

/** Returned by tagTwrLocalize_TC — the tag has all ranges after one full cycle. */
typedef struct RangeInfrastructureResult_TC {
    boolean  success;
    uint8_t  num_anchors;                    // how many anchors were ranged
    uint16_t anchor_ids[TC_MAX_ANCHORS];     // short addresses of each anchor
    double   ranges[TC_MAX_ANCHORS];         // corresponding distances in metres
    uint32_t new_blink_rate;                 // advisory rate sent by the last anchor (0 = unchanged)
} RangeInfrastructureResult_TC;


namespace DW1000NgRTLS_TC {

    /**
     * ANCHOR SIDE
     * Call this inside the anchor's loop().  The anchor listens for a Poll,
     * sends a Response, listens for the Final, then transmits either a
     * RANGING_CONFIRM (point the tag to the next anchor) or ACTIVITY_FINISHED
     * (tell the tag the cycle is done and optionally supply a new blink rate).
     *
     * @param next   NextActivity::RANGING_CONFIRM      supply next anchor address in `value`
     *               NextActivity::ACTIVITY_FINISHED    supply new blink rate in `value`
     * @param value  Interpretation depends on `next` (see above).
     */
    RangeAcceptResult_TC anchorRangeAccept(NextActivity next, uint16_t value);

    /**
     * TAG SIDE — single anchor
     * Range against one anchor whose short address is `anchor_address`.
     * `reply_delay_us` is the tag-side delay before transmitting the Final
     * message (1500 µs works for 8–80 MHz devices; decrease to improve
     * throughput once the system is stable).
     *
     * Returns a RangeResult (defined in DW1000NgRTLS.hpp):
     *   .success           exchange completed without timeout
     *   .next              anchor signalled RANGING_CONFIRM (more anchors to go)
     *   .next_anchor       short address of the next anchor (valid when .next == true)
     *   .new_blink_rate    advisory rate from anchor (valid when .next == false)
     */
    RangeResult tagRangeOne(uint16_t anchor_address, uint16_t reply_delay_us);

    /**
     * TAG SIDE — full localization cycle
     * Ranges against every anchor in the `anchor_addresses` array (length
     * `num_anchors`), collects all distances, then returns them together in
     * a single RangeInfrastructureResult_TC.
     *
     * The function stops early and marks success=false if any single exchange
     * fails.  Gaps between polls are governed by `reply_delay_us`.
     */
    RangeInfrastructureResult_TC tagTwrLocalize(
        uint16_t* anchor_addresses,
        uint8_t   num_anchors,
        uint16_t  reply_delay_us
    );

} // namespace DW1000NgRTLS_TC
