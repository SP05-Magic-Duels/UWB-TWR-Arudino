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

// SPI porting pins
#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define SPI_CS 4

// Connection pins
const uint8_t PIN_RST = 27; // reset pin
const uint8_t PIN_IRQ = 34; // irq pin
const uint8_t PIN_SS  = 4;  // spi select pin

/**
 * Message utility functions
 */

#define POLL 0
#define POLL_ACK 1
#define RANGE 2
#define RANGE_REPORT 3
#define RANGE_FAILED 255

// Message flow state
volatile byte _expectedMsgId;

// Message sent/received state
volatile boolean _sentAck = false;
volatile boolean _receivedAck = false;
// Protocol error state
boolean _protocolFailed = false;

// Timestamps to remember
uint64_t _timePollSent;
uint64_t _timePollReceived;
uint64_t _timePollAckSent;
uint64_t _timePollAckReceived;
uint64_t _timeRangeSent;
uint64_t _timeRangeReceived;

// Data buffer
#define LEN_DATA 16
byte _data[LEN_DATA];

// Watchdog and reset period
uint32_t _lastActivity;
uint32_t _resetPeriod = 250;

// Reply times (same on both sides for symm. ranging)
uint16_t _replyDelayTimeUS = 3000;

// Last computed range/time
uint64_t _timeComputedRange;

device_configuration_t DEFAULT_CONFIG = {
    false,
    true,
    true,
    true,
    false,
    SFDMode::STANDARD_SFD,
    Channel::CHANNEL_5,
    DataRate::RATE_850KBPS,
    PulseFrequency::FREQ_16MHZ,
    PreambleLength::LEN_256,
    PreambleCode::CODE_3
};

interrupt_configuration_t DEFAULT_INTERRUPT_CONFIG = {
    true,
    true,
    true,
    false,
    true
};

sleep_configuration_t SLEEP_CONFIG = {
    false,  // onWakeUpRunADC   reg 0x2C:00
    false,  // onWakeUpReceive
    false,  // onWakeUpLoadEUI
    true,   // onWakeUpLoadL64Param
    true,   // preserveSleep
    true,   // enableSLP    reg 0x2C:06
    false,  // enableWakePIN
    true    // enableWakeSPI
};

frame_filtering_configuration_t TAG_FRAME_FILTER_CONFIG = {
    false,
    false,
    true,
    false,
    false,
    false,
    false,
    false
};

frame_filtering_configuration_t ANCHOR_FRAME_FILTER_CONFIG = {
    false,
    false,
    true,
    false,
    false,
    false,
    false,
    true        /* This allows blink frames */
};