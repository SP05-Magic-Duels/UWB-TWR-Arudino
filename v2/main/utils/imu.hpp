#include "FastIMU.h"
#include <Wire.h>

#define IMU_ADDRESS 0x68
#define PERFORM_CALIBRATION
BMI160 IMU;

calData calib = { 0 };  // Calibration data
AccelData accelData;    // Sensor data
GyroData gyroData;

void app_setup() {
	Serial.begin(115200);
	delay(1000); // Wait for Serial to initialize

	Wire.setPins(21, 22);
	Wire.begin();
	Wire.setClock(400000); // 400khz clock

	int err = IMU.init(calib, IMU_ADDRESS);
	if (err != 0) {
		Serial.print("Error initializing IMU: ");
		Serial.println(err);
		while (true) {
		;
		}
	}
	
#ifdef PERFORM_CALIBRATION
	Serial.println("FastIMU calibration & data example");

	delay(5000);
	Serial.println("Keep IMU level.");
	delay(5000);
	IMU.calibrateAccelGyro(&calib);
	Serial.println("Calibration done!");
	Serial.println("Accel biases X/Y/Z: ");
	Serial.print(calib.accelBias[0]);
	Serial.print(",");
	Serial.print(calib.accelBias[1]);
	Serial.print(",");
	Serial.println(calib.accelBias[2]);
	Serial.println("Gyro biases X/Y/Z: ");
	Serial.print(calib.gyroBias[0]);
	Serial.print(",");
	Serial.print(calib.gyroBias[1]);
	Serial.print(",");
	Serial.println(calib.gyroBias[2]);

	delay(5000);
	IMU.init(calib, IMU_ADDRESS);
#endif

	//err = IMU.setGyroRange(500);      //USE THESE TO SET THE RANGE, IF AN INVALID RANGE IS SET IT WILL RETURN -1
	//err = IMU.setAccelRange(2);       //THESE TWO SET THE GYRO RANGE TO ±500 DPS AND THE ACCELEROMETER RANGE TO ±2g
	
	if (err != 0) {
		Serial.print("Error Setting range: ");
		Serial.println(err);
		while (true) {
		;
		}
	}
}

void app_loop() {
	IMU.update();
	IMU.getAccel(&accelData);
	Serial.print(accelData.accelX);
	// Serial.print("\t");
	Serial.print(", ");
	Serial.print(accelData.accelY);
	// Serial.print("\t");
	Serial.print(", ");
	Serial.print(accelData.accelZ);
	// Serial.print("\t");
	Serial.print(", ");

	IMU.getGyro(&gyroData);
	Serial.print(gyroData.gyroX);
	// Serial.print("\t");
	Serial.print(", ");
	Serial.print(gyroData.gyroY);
	// Serial.print("\t");
	Serial.print(", ");
	Serial.print(gyroData.gyroZ);
	Serial.println();

	// if (IMU.hasTemperature()) {
	// 	Serial.print("\t");
	// 	Serial.println(IMU.getTemp());
	// }
	// else {
	// 	Serial.println();
	// }
	delay(50);
}
