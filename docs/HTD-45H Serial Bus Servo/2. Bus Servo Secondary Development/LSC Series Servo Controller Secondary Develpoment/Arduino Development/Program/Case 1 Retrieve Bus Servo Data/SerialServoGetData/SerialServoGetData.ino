#include "include.h"
#include "SerialServo.h"

void setup() {
  Serial.begin(115200);
  delay(1000);
}

void loop() {
  int id = LobotSerialServoReadID(Serial);  //读取当前舵机ID
  int pos = LobotSerialServoReadPosition(Serial, id); //读取舵机实时位置
  int dev = LobotSerialServoReadDev(Serial, id);  //读取舵机偏差，默认0
  LobotSerialServoReadAngleRange(Serial, id);   //读取舵机角度限位，默认（0-1000）
  LobotSerialServoReadVinLimit(Serial, id); //读取电压限制范围，默认（4.50-14.00V）
  int temperature_warn = LobotSerialServoReadTempLimit(Serial, id); //读取温度报警阈值，默认（85°）
  int temperature = LobotSerialServoReadTemp(Serial, id); //读取舵机实时温度
  int vin = LobotSerialServoReadVin(Serial, id);  //读取舵机实时电压
  int lock = LobotSerialServoReadLoadOrUnload(Serial, id);  //读取舵机状态
  Serial.println("");
  Serial.print("id:");
  Serial.println(id);
  Serial.print("pos:");
  Serial.println(pos);
  Serial.print("dev:");
  Serial.println(dev);
  Serial.print("angle_range:");
  Serial.print(retL);
  Serial.print("-");
  Serial.println(retH);
  Serial.print("vin_range:");
  Serial.print((float)vinL/1000);
  Serial.print("-");
  Serial.print((float)vinH/1000);
  Serial.println(" V");
  Serial.print("temperature_warn:");
  Serial.println(temperature_warn);
  Serial.print("temperature:");
  Serial.println(temperature);
  Serial.print("vin:");
  Serial.println((float)vin/1000);
  Serial.print("lock:");
  Serial.println(lock);
  delay(1000);
}
