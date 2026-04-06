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

#include <Arduino.h>
#include "DW1000Ng.hpp"
#include "DW1000NgUtils.hpp"
#include "DW1000NgTime.hpp"
#include "DW1000NgRanging.hpp"
#include "DW1000NgRTLS.hpp"
#include "DW1000NgRTLSTC.hpp"

namespace DW1000NgRTLS_TC {

RangeAcceptResult_TC anchorRangeAccept(NextActivity next, uint16_t /*value_unused*/) {

    // Step 1: wait for Poll
    if (!DW1000NgRTLS::receiveFrame()) {
        return {false, 0.0, 0};
    }

    size_t poll_len = DW1000Ng::getReceivedDataLength();
    byte   poll_data[poll_len];
    DW1000Ng::getReceivedData(poll_data, poll_len);

    if (poll_len <= 9 || poll_data[9] != RANGING_TAG_POLL) {
        return {false, 0.0, 0};
    }

    uint64_t timePollReceived = DW1000Ng::getReceiveTimestamp();
    uint16_t source_address =
        static_cast<uint16_t>(DW1000NgUtils::bytesAsValue(&poll_data[7], 2));

    // Step 2: send Response-to-Poll
    DW1000NgRTLS::transmitResponseToPoll(&poll_data[7]);
    DW1000NgRTLS::waitForTransmission();
    uint64_t timeResponseToPollSent = DW1000Ng::getTransmitTimestamp();

    delayMicroseconds(1500); // must match tag-side FINAL_MSG_DELAY_US

    // Step 3: wait for Final
    if (!DW1000NgRTLS::receiveFrame()) {
        return {false, 0.0, 0};
    }

    size_t final_len = DW1000Ng::getReceivedDataLength();
    byte   final_data[final_len];
    DW1000Ng::getReceivedData(final_data, final_len);

    if (final_len <= 18 || final_data[9] != RANGING_TAG_FINAL_RESPONSE_EMBEDDED) {
        return {false, 0.0, 0};
    }

    uint64_t timeFinalReceived = DW1000Ng::getReceiveTimestamp();

    // Step 4: compute range (asymmetric TWR)
    double range = DW1000NgRanging::computeRangeAsymmetric(
        DW1000NgUtils::bytesAsValue(final_data + 10, LENGTH_TIMESTAMP), // T_sp
        timePollReceived,                                                // T_rp
        timeResponseToPollSent,                                          // T_sr
        DW1000NgUtils::bytesAsValue(final_data + 14, LENGTH_TIMESTAMP), // T_rr
        DW1000NgUtils::bytesAsValue(final_data + 18, LENGTH_TIMESTAMP), // T_sf
        timeFinalReceived                                                // T_rf
    );

    range = DW1000NgRanging::correctRange(range);
    if (range <= 0.0) range = 0.000001;

    // Pack range as uint16_t millimetres (max ~65.5 m — fine for indoor RTLS).
    uint16_t range_mm = static_cast<uint16_t>(
        constrain(range * 1000.0, 1.0, 65535.0)
    );

    // Step 5: transmit control frame with range_mm in bytes 11–12
    byte rangeBytes[2];
    DW1000NgUtils::writeValueToBytes(rangeBytes, range_mm, 2);

    if (next == NextActivity::RANGING_CONFIRM) {
        // transmitRangingConfirm puts rangeBytes into bytes 11–12.
        DW1000NgRTLS::transmitRangingConfirm(&final_data[7], rangeBytes);
    } else {
        // transmitActivityFinished puts rangeBytes into bytes 11–12.
        DW1000NgRTLS::transmitActivityFinished(&final_data[7], rangeBytes);
    }
    DW1000NgRTLS::waitForTransmission();

    return {true, range, source_address};
}


RangeResult tagRangeOne(uint16_t anchor_address, uint16_t reply_delay_us) {
    byte target[2];
    DW1000NgUtils::writeValueToBytes(target, anchor_address, 2);

    // Poll
    DW1000NgRTLS::transmitPoll(target);

    if (!DW1000NgRTLS::waitForNextRangingStep()) {
        return {false, false, 0, 0};
    }

    size_t cont_len = DW1000Ng::getReceivedDataLength();
    byte   cont_recv[cont_len];
    DW1000Ng::getReceivedData(cont_recv, cont_len);

    if (cont_len <= 10 ||
        cont_recv[9]  != ACTIVITY_CONTROL ||
        cont_recv[10] != RANGING_CONTINUE) {
        return {false, false, 0, 0};
    }

    // Final
    DW1000NgRTLS::transmitFinalMessage(
        &cont_recv[7],
        reply_delay_us,
        DW1000Ng::getTransmitTimestamp(), // Poll Tx timestamp
        DW1000Ng::getReceiveTimestamp()   // Response Rx timestamp
    );

    if (!DW1000NgRTLS::waitForNextRangingStep()) {
        return {false, false, 0, 0};
    }

    size_t act_len = DW1000Ng::getReceivedDataLength();
    byte   act_recv[act_len];
    DW1000Ng::getReceivedData(act_recv, act_len);

    if (act_len <= 10 || act_recv[9] != ACTIVITY_CONTROL) {
        return {false, false, 0, 0};
    }

    if (act_len > 12 && act_recv[10] == RANGING_CONFIRM) {
        // Bytes 11–12 carry range_mm packed by the anchor.
        // We reuse the next_anchor field of RangeResult to carry range_mm back
        // to tagTwrLocalize; the tag knows the actual next anchor from its list.
        uint16_t range_mm =
            static_cast<uint16_t>(DW1000NgUtils::bytesAsValue(&act_recv[11], 2));
        return {true, true, range_mm, 0};
    }

    if (act_len > 12 && act_recv[10] == ACTIVITY_FINISHED) {
        // Bytes 11–12 carry range_mm packed by the anchor.
        // We reuse new_blink_rate to carry range_mm back to tagTwrLocalize.
        uint16_t range_mm =
            static_cast<uint16_t>(DW1000NgUtils::bytesAsValue(&act_recv[11], 2));
        return {true, false, 0, static_cast<uint32_t>(range_mm)};
    }

    return {false, false, 0, 0};
}

// Tag side: full localization cycle
RangeInfrastructureResult_TC tagTwrLocalize(
    uint16_t* anchor_addresses,
    uint8_t   num_anchors,
    uint16_t  reply_delay_us
) {
    RangeInfrastructureResult_TC result;
    result.success        = false;
    result.num_anchors    = 0;
    result.new_blink_rate = 0;
    memset(result.anchor_ids, 0, sizeof(result.anchor_ids));
    memset(result.ranges,     0, sizeof(result.ranges));

    if (num_anchors == 0 || num_anchors > TC_MAX_ANCHORS) {
        return result;
    }

    for (uint8_t i = 0; i < num_anchors; i++) {
        bool is_last = (i == num_anchors - 1);

        RangeResult r = tagRangeOne(anchor_addresses[i], reply_delay_us);

        if (!r.success) {
            return result; // abort the whole cycle on any failure
        }

        // Decode range_mm from the appropriate RangeResult field.
        double decoded_range;
        if (!is_last) {
            // RANGING_CONFIRM: range_mm was packed into r.next_anchor.
            decoded_range = static_cast<double>(r.next_anchor) / 1000.0;
        } else {
            // ACTIVITY_FINISHED: range_mm was packed into r.new_blink_rate.
            decoded_range = static_cast<double>(r.new_blink_rate) / 1000.0;
        }

        result.anchor_ids[result.num_anchors] = anchor_addresses[i];
        result.ranges[result.num_anchors]     = decoded_range;
        result.num_anchors++;

        if (!is_last) {
            yield();
            delayMicroseconds(500); // brief gap before polling the next anchor
        }
    }

    result.success = (result.num_anchors == num_anchors);
    return result;
}

} // namespace DW1000NgRTLS_TC
