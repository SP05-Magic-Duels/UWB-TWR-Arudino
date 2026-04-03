
// #include <esp_now.h>
// #include <WiFi.h>
// #include <esp_wifi.h>

// typedef struct struct_message {
// 	float x;
// 	float y;
// 	float z;
// } struct_message;

// struct_message data;
// esp_now_peer_info_t ESPNOW_peerInfo = {};

// // FreeRTOS handles
// TaskHandle_t  espnowTaskHandle = NULL;
// SemaphoreHandle_t dataMutex    = NULL;

// void _espnowOnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
// 	// Serial.print("\nSend Status: ");
// 	// Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");

// 	// For debugging
// 	// Serial.printf("To: %02X:%02X:%02X:%02X:%02X:%02X | XYZ: %.2f, %.2f, %.2f\n",
// 	//             targetAddress[0], targetAddress[1], targetAddress[2],   // Fix later
// 	//             targetAddress[3], targetAddress[4], targetAddress[5],   // Fix later
// 	//             data.x, data.y, data.z);
// }

// void _espnowTask(void *pvParameters) {
// 	struct_message localData;

// 	for (;;) {
// 		// Safely copy the latest position from Core 1
// 		if (xSemaphoreTake(dataMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
// 		localData = data;
// 		xSemaphoreGive(dataMutex);
// 		}

// 		esp_now_send(targetAddress, (uint8_t *)&localData, sizeof(localData));

// 		vTaskDelay(pdMS_TO_TICKS(100));  // Send at 10 Hz — adjust as needed
// 	}
// }

// void _espnowInit() {
// 	WiFi.mode(WIFI_STA);
// 	WiFi.begin();
	
// 	while (WiFi.macAddress() == "00:00:00:00:00:00") {
// 		delay(10);
// 	}
	
// 	// Force same channel
// 	esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

// 	Serial.print("My MAC: ");
// 	Serial.println(WiFi.macAddress());

// 	if (esp_now_init() != ESP_OK) {
// 		Serial.println("Error initializing ESP-NOW");
// 		return;
// 	}

// 	esp_now_register_send_cb(esp_now_send_cb_t(_espnowOnDataSent));
	
// 	memcpy(ESPNOW_peerInfo.peer_addr, targetAddress, 6);
// 	ESPNOW_peerInfo.channel = 0; // Use current channel
// 	ESPNOW_peerInfo.encrypt = false;
	
// 	if (esp_now_add_peer(&ESPNOW_peerInfo) != ESP_OK) {
// 		Serial.println("Failed to add peer");
// 	}

// 	// Register task for sending data over ESPNOW
// 	// Create mutex to protect shared data between cores
// 	dataMutex = xSemaphoreCreateMutex();
// 	if (dataMutex == NULL) {
// 		Serial.println("Failed to create mutex");
// 		return;
// 	}

// 	// Spawn ESP-NOW task on Core 0
// 	// Arduino loop() and DW1000 run on Core 1 — they never share a core
// 	xTaskCreatePinnedToCore(
// 		_espnowTask,       // Task function
// 		"_espnowTask",     // Name (for debugging)
// 		4096,              // Stack size (bytes)
// 		NULL,              // Parameters
// 		1,                 // Priority (1 = low, leaves Core 0 headroom)
// 		&espnowTaskHandle, // Handle
// 		0                  // Core 0
// 	);

// 	Serial.println("ESP-NOW task started on Core 0");
// }
