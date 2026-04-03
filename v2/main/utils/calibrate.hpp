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
 * Copyright (c) 2015 by Thomas Trojer <thomas@trojer.net>
 * Decawave DW1000 library for arduino.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <DW1000Ng.hpp>
#include <DW1000NgUtils.hpp>
#include <DW1000NgRanging.hpp>

#include "configs.hpp"
#include "polling.hpp"

#define EXPECTED_RANGE 7.94 // Recommended value for default values, refer to chapter 8.3.1 of DW1000 User manual
#define EXPECTED_RANGE_EPSILON 0.05
#define ACCURACY_THRESHOLD 5
#define ANTENNA_DELAY_STEPS 1

// Message flow state

// Antenna calibration variables
int _accuracyCounter = 0;
uint16_t _antenna_delay = 16436;

// Ranging counter (per second)
uint16_t _successRangingCount = 0;
uint32_t _rangingCountPeriod = 0;
float _samplingRate = 0;

volatile boolean _calibrationComplete = false;

void _calibrate() {
    Serial.print(F("_sendAck: ")); Serial.println(_sentAck);
    Serial.print(F("_receivedAck: ")); Serial.println(_receivedAck);
    Serial.print(F("_expectedMsgId: ")); Serial.println(_expectedMsgId);
    Serial.print(F("_antenna_delay: ")); Serial.println(_antenna_delay);

    Serial.print("IRQ status: "); Serial.println(digitalRead(PIN_IRQ));
    Serial.println();

    int32_t curMillis = millis();
    if (!_sentAck && !_receivedAck) {
        // check if inactive
        if (curMillis - _lastActivity > _resetPeriod) {
            _resetInactive();
        }
        return;
    }

    // Continue on any success confirmation
    if (_sentAck) {
        _sentAck = false;
        byte msgId = _data[0];
        if (msgId == POLL_ACK) {
            _timePollAckSent = DW1000Ng::getTransmitTimestamp();
            _noteActivity();
        }
        DW1000Ng::startReceive();
    }
    
    if (_receivedAck) {
        _receivedAck = false;
        // get message and parse
        DW1000Ng::getReceivedData(_data, LEN_DATA);
        byte msgId = _data[0];
        if (msgId != _expectedMsgId) {
            // Unexpected message, start over again (except if already POLL)
            _protocolFailed = true;
        }
        if (msgId == POLL) {
            // On POLL we (re-)start, so no protocol failure
            _protocolFailed = false;
            _timePollReceived = DW1000Ng::getReceiveTimestamp();
            _expectedMsgId = RANGE;
            _transmitPollAck();
            _noteActivity();
        }
        else if (msgId == RANGE) {
            _timeRangeReceived = DW1000Ng::getReceiveTimestamp();
            _expectedMsgId = POLL;
            if (!_protocolFailed) {
                _timePollSent = DW1000NgUtils::bytesAsValue(_data + 1, LENGTH_TIMESTAMP);
                _timePollAckReceived = DW1000NgUtils::bytesAsValue(_data + 6, LENGTH_TIMESTAMP);
                _timeRangeSent = DW1000NgUtils::bytesAsValue(_data + 11, LENGTH_TIMESTAMP);
                // (re-)compute range as two-way ranging is done
                double distance = DW1000NgRanging::computeRangeAsymmetric(_timePollSent,
                                                            _timePollReceived, 
                                                            _timePollAckSent, 
                                                            _timePollAckReceived, 
                                                            _timeRangeSent, 
                                                            _timeRangeReceived);
                
                distance = DW1000NgRanging::correctRange(distance);
                
                String rangeString = "Range: "; rangeString += distance; rangeString += " m";
                rangeString += "\t RX power: "; rangeString += DW1000Ng::getReceivePower(); rangeString += " dBm";
                rangeString += "\t Sampling: "; rangeString += _samplingRate; rangeString += " Hz";
                Serial.println(rangeString);

                // Antenna delay script
                if(distance >= (EXPECTED_RANGE - EXPECTED_RANGE_EPSILON) && distance <= (EXPECTED_RANGE + EXPECTED_RANGE_EPSILON)) {
                    _accuracyCounter++;
                } 
                else {
                    _accuracyCounter = 0;
                    _antenna_delay += (distance > EXPECTED_RANGE) ? ANTENNA_DELAY_STEPS : -ANTENNA_DELAY_STEPS;
                    DW1000Ng::setAntennaDelay(_antenna_delay);
                }

                if(_accuracyCounter == ACCURACY_THRESHOLD) {
                    Serial.print("Found Antenna Delay value (Divide by two if one antenna is set to 0): ");
                    Serial.println(_antenna_delay);
                    _calibrationComplete = true;
                    delay(10000);
                }

                // Serial.print("FP power is [dBm]: "); Serial.print(DW1000Ng::getFirstPathPower());
                // Serial.print("RX power is [dBm]: "); Serial.println(DW1000Ng::getReceivePower());
                // Serial.print("Receive quality: "); Serial.println(DW1000Ng::getReceiveQuality());
                // Update sampling rate (each second)
                _transmitRangeReport(distance * DISTANCE_OF_RADIO_INV);
                _successRangingCount++;
                if (curMillis - _rangingCountPeriod > 1000) {
                    _samplingRate = (1000.0f * _successRangingCount) / (curMillis - _rangingCountPeriod);
                    _rangingCountPeriod = curMillis;
                    _successRangingCount = 0;
                }
            }
            else {
                _transmitRangeFailed();
            }

            _noteActivity();
        }
    }
}
