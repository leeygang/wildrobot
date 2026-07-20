/**
 * @file HX_30HM.h
 * @author Min
 * @brief Servo Control Interface Function Implementation Example
 * @version 1.0
 * @date 2025-10-21
 *
 * @copyright Copyright (c) 2025
 */
#include "HX_30HM.h"

#define RXD1 16
#define TXD1 17

// Create a SerialServo instance using the Serial port with a baud rate of 115200, TX pin 1, RX pin 0 (创建SerialServo实例，使用Serial串口，波特率为115200，TX引脚1，RX引脚0)
// Note: Modify according to the actual serial port used (e.g. Serial, Serial1, Serial2, etc.)(注意：根据实际连接的串口修改（可以是Serial, Serial1, Serial2等）)
// Modify TX and RX pin numbers according to the actual hardware connection(根据实际硬件连接修改TX和RX引脚号)
SerialServo servo(Serial1, 1000000, TXD1, RXD1);

ServoStatus_t status;

int16_t write_pos = 4096;
int16_t read_pos = 4096;

int16_t write_pos_offset = 100;
int16_t read_pos_offset;

uint8_t write_acc = 100;

int16_t write_speed = 1000;
int16_t read_speed;

int16_t write_pwm_speed = 1000;

uint16_t write_torque = 1000;

int16_t sync_write1[2][4] = {{1, 0, 1000, 4095},
							               {2, 0, 1000, 4095}};

int16_t sync_write2[2][4] = {{1, 0, 1000, 0},
							               {2, 0, 1000, 0}};
uint8_t read_id[] = {1, 2};
int16_t sync_read_data[2][5];

uint8_t temp;
uint8_t vol;
uint16_t cur;
int16_t read_load;
uint8_t moving_status;

void setup() {
  Serial.begin(115200);

  status = servo.ping(0xFE);    //Broadcast search for servos (广播搜索舵机)
  // Serial.printf("ID:%d\n", status.id);  //Print received servo ID (接收到的 ID 号)
  // Serial.printf("状态：%X\n", status.error_byte);    //Print servo working status if a data frame is received, see ServoStatus_t in HX_30HM.h(该舵机的工作状态(若接收到数据帧), 参考HX_30HM.h中 ServoStatus_t 结构体注释)
  
  // status = servo.enable_torque(1);         // Enable servo torque output(启用1号的舵机的力矩输出 )
  // status = servo.disable_torque(1);         // Disable servo torque output (禁用1号的舵机的力矩输出)
  // status = servo.cali_pos(1);              // Calibrate servo position(校准1号的当前舵机位置 )
  //status = servo.select_mode(1, 0);       // Select servo working mode (选择1号的舵机工作模式)

  // status = servo.write_pos(1, write_pos);   // Set servo target position (设置1号的舵机目标位置)
  // status = servo.read_pos(1, &read_pos);    // Read servo current position value (读取1号的舵机当前位置值)
  // Serial.println(read_pos);

  // status = servo.write_pos_offset(1, write_pos_offset);  // Set servo position offset (设置1号的舵机位置偏差)
	// status = servo.read_pos_offset(1, &read_pos_offset);   // Read servo position offset (读取1号的舵机位置偏差)
  // Serial.println(read_pos_offset);

  // status = servo.write_acc(1, write_acc);   // Set servo acceleration parameter (设置1号的舵机加速度参数)

  // status = servo.write_speed(1, write_speed); // Set servo target speed (设置1号的舵机目标速度)
	// status = servo.read_speed(1, &read_speed);  // Read servo current speed (读取1号的舵机当前速度)  
  // Serial.println(read_speed);

  // status = servo.read_pos_speed(1, &read_pos, &read_speed); // 同时读取1号的舵机位置和速度 (Read servo position and speed)
  // Serial.print(read_pos);
  // Serial.print(", ");
  // Serial.println(read_speed);

  // status = servo.write_pos_ex(1, write_acc, write_speed, write_pos); // Extended position control: set servo acceleration, speed, and position simultaneously (扩展位置控制：同时设置1号的舵机加速度、速度和位置)
  // status = servo.write_pwm_speed(1, write_pwm_speed);     // Set servo PWM speed (effective in PWM mode) (设置1号的舵机PWM速度（PWM模式下生效）)
	// status = servo.write_max_torque(1, write_torque);       // Set servo maximum torque limit (设置1号的舵机最大力矩限制)
  // status = servo.write_reg_pos_ex(1, write_acc, write_speed, 0); // Write acceleration, speed, and position information to servo register address (向1号舵机指定寄存器地址写入加速度、速度和位置信息（暂不执行，需调用action触发）)
  // status = servo.reg_action(1);               // Execute servo action instruction (执行1号的舵机动作指令)
  // status = servo.sync_write_pos_ex(sync_write1, 2);      // Write acceleration, speed, and position information to servo 1 and 2 simultaneously (向1号和2号舵机同时写入加速度、速度和位置信息)
  // status = servo.sync_read_cur_pos_ex(read_id, 2, sync_read_data);  // Read acceleration, speed, and position information from servo 1 and 2 simultaneously (同部读取1号和2号舵机加速度、速度和位置信息)
  // for(uint8_t i = 0; i < 2; i++) {
  //   for(uint8_t j = 0; j < 5; j++) {
  //     Serial.print(sync_read_data[i][j]);
  //     Serial.print(", ");
  //   }
  //   Serial.println();
  // }
	// status = servo.read_temperture(1, &temp);   // Read servo temperature value (读取1号舵机温度值)
  // Serial.println(temp);
	// status = servo.read_voltage(1, &vol);       //  Read servo voltage value (读取1号舵机电压值)
  // Serial.println(vol);
	// status = servo.read_current(1, &cur);       //  Read servo current value (读取1号舵机电流值)
  // Serial.println(cur);
	// status = servo.read_load(1, &read_load);    // Read servo load value (读取1号舵机负载值)
  // Serial.println(read_load);
	// status = servo.read_moving_status(1, &moving_status); //  Read servo moving status (读取1号舵机运动状态)
  // Serial.println(moving_status);
}

void loop() {

}
