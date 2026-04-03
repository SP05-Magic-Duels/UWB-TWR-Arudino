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

#pragma once

#include <Arduino.h>
#include "DW1000NgRTLS.hpp"

#define MAX_ANCHORS 8

typedef struct RangeResult_TC {
    boolean success;
    boolean next;
    uint16_t next_anchor;
    uint32_t new_blink_rate;
    double calculatedRange; // Range that is returned by the anchor
} RangeResult_TC;

typedef struct RangeInfrastructureResult_TC {
    boolean success;
    uint16_t new_blink_rate;
    double ranges[MAX_ANCHORS];        // Hold distances for up to MAX_ANCHORS anchors
    uint16_t anchor_ids[MAX_ANCHORS];  // Track which anchor provided which range
} RangeInfrastructureResult_TC;

typedef struct RangeAcceptResult_TC {
    boolean success;
    double range;
    uint16_t source_address;
} RangeAcceptResult_TC;

namespace DW1000NgRTLS_TC {

    // Modified to accept range parameter
    void transmitRangingConfirm(byte tag_short_address[], byte next_anchor[], double range);
    
    // Modified to accept range parameter
    void transmitActivityFinished(byte tag_short_address[], byte blink_rate[], double range);

    /* Used by an anchor to accept an incoming tagRangeRequest by means of the infrastructure
       NextActivity is used to indicate the tag what to do next after the ranging process (Activity finished is to return to blink (range request), 
        Continue range is to tell the tag to range a new anchor)
       value is the value relative to the next activity (Activity finished = new blink rante, continue range = new anchor address)
    */
    RangeAcceptResult_TC anchorRangeAccept(NextActivity next, uint16_t value);

    /* Used by tag to range after range request accept of the infrastructure 
       Target anchor is given after a range request success
       Finalmessagedelay is used in the process of TWR, a value of 1500 works on 8mhz-80mhz range devices,
        you could try to decrease it to improve system performance.
    */
    RangeInfrastructureResult_TC tagRangeInfrastructure(uint16_t target_anchor, uint16_t finalMessageDelay);

    /* Can be used as a single function start the localization process from the tag.
        Finalmessagedelay is the same as in function tagRangeInfrastructure
    */
    RangeInfrastructureResult_TC tagTwrLocalize(uint16_t finalMessageDelay);
}