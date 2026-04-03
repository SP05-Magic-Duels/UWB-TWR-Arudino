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
#include <DW1000NgTime.hpp>
#include <DW1000NgConstants.hpp>

#include "configs.hpp"

void _printDeviceInfo() {
    char msg[128];
    DW1000Ng::getPrintableDeviceIdentifier(msg);
    Serial.print("Device ID: "); Serial.println(msg);
    
    DW1000Ng::getPrintableExtendedUniqueIdentifier(msg);
    Serial.print("Unique ID: "); Serial.println(msg);
    
    DW1000Ng::getPrintableNetworkIdAndShortAddress(msg);
    Serial.print("Network ID & Device Address: "); Serial.println(msg);
    
    DW1000Ng::getPrintableDeviceMode(msg);
    Serial.print("Device mode: "); Serial.println(msg);
}

void _receiver() {
    DW1000Ng::forceTRxOff();
    // So we don't need to restart the receiver manually
    DW1000Ng::startReceive();
}

void _noteActivity() {
    // Update activity timestamp, so that we do not reach "resetPeriod"
    _lastActivity = millis();
}

void _handleSent() {
    // Status change on sent success
    _sentAck = true;
}

void _handleReceived() {
    // Status change on received success
    _receivedAck = true;
}

void _transmitPoll() {
    _data[0] = POLL;
    DW1000Ng::setTransmitData(_data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void _transmitPollAck() {
    _data[0] = POLL_ACK;
    DW1000Ng::setTransmitData(_data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void _resetInactive() {
    // Tag sends POLL and listens for POLL_ACK
    _expectedMsgId = POLL_ACK;
    DW1000Ng::forceTRxOff();
    _transmitPoll();
    _noteActivity();
}

void _resetInactiveAnchor() {
    _expectedMsgId = POLL;
    _receiver();
    _noteActivity();
}

void _transmitRangeReport(float curRange) {
    _data[0] = RANGE_REPORT;
    // write final ranging result
    memcpy(_data + 1, &curRange, 4);
    DW1000Ng::setTransmitData(_data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void _transmitRangeFailed() {
    _data[0] = RANGE_FAILED;
    DW1000Ng::setTransmitData(_data, LEN_DATA);
    DW1000Ng::startTransmit();
}

void _transmitRange() {
    _data[0] = RANGE;

    // Calculation of future time
    byte futureTimeBytes[LENGTH_TIMESTAMP];

	_timeRangeSent = DW1000Ng::getSystemTimestamp();
	_timeRangeSent += DW1000NgTime::microsecondsToUWBTime(_replyDelayTimeUS);
    DW1000NgUtils::writeValueToBytes(futureTimeBytes, _timeRangeSent, LENGTH_TIMESTAMP);
    DW1000Ng::setDelayedTRX(futureTimeBytes);
    _timeRangeSent += DW1000Ng::getTxAntennaDelay();

    DW1000NgUtils::writeValueToBytes(_data + 1, _timePollSent, LENGTH_TIMESTAMP);
    DW1000NgUtils::writeValueToBytes(_data + 6, _timePollAckReceived, LENGTH_TIMESTAMP);
    DW1000NgUtils::writeValueToBytes(_data + 11, _timeRangeSent, LENGTH_TIMESTAMP);
    DW1000Ng::setTransmitData(_data, LEN_DATA);
    DW1000Ng::startTransmit(TransmitMode::DELAYED);
    // Serial.print("Expect RANGE to be sent @ "); Serial.println(timeRangeSent.getAsFloat());
}
