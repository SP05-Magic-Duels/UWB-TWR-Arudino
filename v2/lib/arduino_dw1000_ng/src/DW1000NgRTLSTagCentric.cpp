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
#include "DW1000NgRTLSTagCentric.hpp"
#include "DW1000Ng.hpp"
#include "DW1000NgUtils.hpp"
#include "DW1000NgTime.hpp"
#include "DW1000NgRanging.hpp"

namespace DW1000NgRTLS_TC {

    void transmitRangingConfirm(byte tag_short_address[], byte next_anchor[], double range) {
        byte rangingConfirm[] = {DATA, SHORT_SRC_AND_DEST, DW1000NgRTLS::increaseSequenceNumber(), 0,0, 0,0, 0,0, ACTIVITY_CONTROL, RANGING_CONFIRM, next_anchor[0], next_anchor[1], 0, 0};
        DW1000Ng::getNetworkId(&rangingConfirm[3]);
        memcpy(&rangingConfirm[5], tag_short_address, 2);
        DW1000Ng::getDeviceAddress(&rangingConfirm[7]);
        
        // Convert range to mm and embed in packet
        uint16_t rangeMm = (uint16_t)(range * 1000);
        DW1000NgUtils::writeValueToBytes(&rangingConfirm[13], rangeMm, 2);
        
        DW1000Ng::setTransmitData(rangingConfirm, sizeof(rangingConfirm));
        DW1000Ng::startTransmit();
    }

    void transmitActivityFinished(byte tag_short_address[], byte blink_rate[], double range) {
        byte rangingConfirm[] = {DATA, SHORT_SRC_AND_DEST, DW1000NgRTLS::increaseSequenceNumber(), 0,0, 0,0, 0,0, ACTIVITY_CONTROL, ACTIVITY_FINISHED, blink_rate[0], blink_rate[1], 0, 0};
        DW1000Ng::getNetworkId(&rangingConfirm[3]);
        memcpy(&rangingConfirm[5], tag_short_address, 2);
        DW1000Ng::getDeviceAddress(&rangingConfirm[7]);
        
        uint16_t rangeMm = (uint16_t)(range * 1000);
        DW1000NgUtils::writeValueToBytes(&rangingConfirm[13], rangeMm, 2);
        
        DW1000Ng::setTransmitData(rangingConfirm, sizeof(rangingConfirm));
        DW1000Ng::startTransmit();
    }

    RangeResult_TC tagFinishRange(uint16_t anchor, uint16_t replyDelayUs) {
        RangeResult_TC returnValue = {false, false, 0, 0, 0.0};

        byte target_anchor[2];
        DW1000NgUtils::writeValueToBytes(target_anchor, anchor, 2);
        DW1000NgRTLS::transmitPoll(target_anchor);
        /* Start of poll control for range */
        if(!DW1000NgRTLS::waitForNextRangingStep()) {
            return returnValue;

        } else {

            size_t cont_len = DW1000Ng::getReceivedDataLength();
            byte cont_recv[cont_len];
            DW1000Ng::getReceivedData(cont_recv, cont_len);

            if (cont_len > 10 && cont_recv[9] == ACTIVITY_CONTROL && cont_recv[10] == RANGING_CONTINUE) {
                /* Received Response to poll */
                DW1000NgRTLS::transmitFinalMessage(
                    &cont_recv[7], 
                    replyDelayUs, 
                    DW1000Ng::getTransmitTimestamp(), // Poll transmit time
                    DW1000Ng::getReceiveTimestamp()  // Response to poll receive time
                );

                if(!DW1000NgRTLS::waitForNextRangingStep()) {
                    returnValue = {false, false, 0, 0, 0.0};
                } else {

                    size_t act_len = DW1000Ng::getReceivedDataLength();
                    byte act_recv[act_len];
                    DW1000Ng::getReceivedData(act_recv, act_len);

                    if(act_len > 10 && act_recv[9] == ACTIVITY_CONTROL) {
                        // Extract range from the last 2 bytes added in step A
                        double dist = 0.0;
                        if (act_len >= 15) {
                            dist = DW1000NgUtils::bytesAsValue(&act_recv[13], 2) / 1000.0;
                        }

                        if (act_len > 12 && act_recv[10] == RANGING_CONFIRM) {
                            returnValue = {true, true, static_cast<uint16_t>(DW1000NgUtils::bytesAsValue(&act_recv[11], 2)), 0, dist};
                        } else if(act_len > 12 && act_recv[10] == ACTIVITY_FINISHED) {
                            returnValue = {true, false, 0, DW1000NgRTLS::calculateNewBlinkRate(act_recv), dist};
                        }
                    } else {
                        returnValue = {false, false, 0, 0, 0.0};
                    }
                }
            } else {
                returnValue = {false, false, 0, 0, 0.0};
            }
            
        }

        return returnValue;
    }

    RangeInfrastructureResult_TC tagRangeInfrastructure(uint16_t target_anchor, uint16_t finalMessageDelay) {
        RangeInfrastructureResult_TC returnValue = {false, 0, {0}, {0}};
        int idx = 0;

        RangeResult_TC result = tagFinishRange(target_anchor, finalMessageDelay);
        uint16_t current_anchor = target_anchor;

        while(result.success && idx < MAX_ANCHORS) {
            returnValue.ranges[idx] = result.calculatedRange;
            returnValue.anchor_ids[idx] = current_anchor;
            idx++;

            if(!result.next) break;

            current_anchor = result.next_anchor;
            result = tagFinishRange(current_anchor, finalMessageDelay);
            yield();
        }

        returnValue.success = (idx > 0);
        return returnValue;
    }

    RangeInfrastructureResult_TC tagTwrLocalize(uint16_t finalMessageDelay) {
        RangeRequestResult request_result = DW1000NgRTLS::tagRangeRequest();

        if(request_result.success) {
            
            RangeInfrastructureResult_TC result = tagRangeInfrastructure(request_result.target_anchor, finalMessageDelay);

            if(result.success)
                return result;
        }
        return {false, 0};
    }

    RangeAcceptResult_TC anchorRangeAccept(NextActivity next, uint16_t value) {
        RangeAcceptResult_TC returnValue = {false, 0, 0};
        double range;

        if(!DW1000NgRTLS::receiveFrame()) {
            return returnValue;

        } else {
            size_t poll_len = DW1000Ng::getReceivedDataLength();
            byte poll_data[poll_len];
            DW1000Ng::getReceivedData(poll_data, poll_len);

            if(poll_len > 9 && poll_data[9] == RANGING_TAG_POLL) {
                uint16_t tagAddr = DW1000NgUtils::bytesAsValue(&poll_data[7], 2);

                uint64_t timePollReceived = DW1000Ng::getReceiveTimestamp();
                DW1000NgRTLS::transmitResponseToPoll(&poll_data[7]);
                DW1000NgRTLS::waitForTransmission();
                uint64_t timeResponseToPoll = DW1000Ng::getTransmitTimestamp();
                delayMicroseconds(1500);

                if(!DW1000NgRTLS::receiveFrame()) {
                    return returnValue;

                } else {
                    size_t rfinal_len = DW1000Ng::getReceivedDataLength();
                    byte rfinal_data[rfinal_len];
                    DW1000Ng::getReceivedData(rfinal_data, rfinal_len);

                    if(rfinal_len > 18 && rfinal_data[9] == RANGING_TAG_FINAL_RESPONSE_EMBEDDED) {
                        uint64_t timeFinalMessageReceive = DW1000Ng::getReceiveTimestamp();

                        range = DW1000NgRanging::computeRangeAsymmetric(
                            DW1000NgUtils::bytesAsValue(rfinal_data + 10, LENGTH_TIMESTAMP), // Poll send time
                            timePollReceived, 
                            timeResponseToPoll, // Response to poll sent time
                            DW1000NgUtils::bytesAsValue(rfinal_data + 14, LENGTH_TIMESTAMP), // Response to Poll Received
                            DW1000NgUtils::bytesAsValue(rfinal_data + 18, LENGTH_TIMESTAMP), // Final Message send time
                            timeFinalMessageReceive // Final message receive time
                        );

                        range = DW1000NgRanging::correctRange(range);

                        /* In case of wrong read due to bad device calibration */
                        if(range <= 0) 
                            range = 0.000001;

                        // Only transmit anchor confirmation packet after adding the range data to response
                        byte finishValue[2];
                        DW1000NgUtils::writeValueToBytes(finishValue, value, 2);

                        if(next == NextActivity::RANGING_CONFIRM) {
                            transmitRangingConfirm(&rfinal_data[7], finishValue, range);
                        } else {
                            transmitActivityFinished(&rfinal_data[7], finishValue, range);
                        }
                        
                        DW1000NgRTLS::waitForTransmission();

                        returnValue = {true, range, tagAddr};
                    }
                }
            }
        }

        return returnValue;
    }

}