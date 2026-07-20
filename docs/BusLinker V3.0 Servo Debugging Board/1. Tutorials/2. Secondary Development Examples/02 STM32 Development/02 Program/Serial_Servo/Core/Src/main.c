/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "HX_30HM.h"
#include "stdio.h"
#include "stdint.h"
#include "Packet_Handler.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define exp 5
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
ServoStatus_t status;

int16_t write_pos = 4096;
int16_t read_pos = 4096;

int16_t write_pos_offset = 100;
int16_t read_pos_offset;

uint8_t write_acc = 100;

int16_t write_speed = 1000;
int16_t read_speed;

int16_t write_pwm_speed = 1000;

uint16_t write_torque = 100;

int16_t sync_write1[2][4] = {{1, 0, 1000, 4095},
							{2, 0, 1000, 4095}};
uint8_t read_id[] = {1, 2};
int16_t sync_read_data[2][5];

uint8_t temp;
uint8_t vol;
uint16_t cur;
int16_t read_load;
uint8_t moving_status;

uint8_t info_buf[255]= {0};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_NVIC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_USART2_UART_Init();

  /* Initialize interrupts */
  MX_NVIC_Init();
  /* USER CODE BEGIN 2 */
  servo_init(); 

#if (exp == 1)
	status = servo_ping(0xFE);
	
	sprintf((char *)info_buf,"TX:%X %X %X %X %X %X\r\n",servo_packet.tx.header_1,\
	                                            servo_packet.tx.header_2,\
																							servo_packet.tx.elements.id,\
																							servo_packet.tx.elements.length,\
																							servo_packet.tx.elements.cmd,\
																							servo_packet.tx.elements.args[servo_packet.tx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"RX:%X %X %X %X %X %X\r\n",servo_packet.rx.header_1,\
	                                            servo_packet.rx.header_2,\
																							servo_packet.rx.elements.id,\
																							servo_packet.rx.elements.length,\
																							servo_packet.rx.elements.cmd,\
																							servo_packet.rx.elements.args[servo_packet.rx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"ID:%X\r\n",status.id);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"statu:%X\r\n",status.error_byte);
	serial_printf_string(info_buf);
	
#elif(exp == 2)
	status = servo_read_pos(1, &read_pos);
	
	sprintf((char *)info_buf,"TX:%X %X %X %X %X %X %X %X\r\n",servo_packet.tx.header_1,\
	                                            servo_packet.tx.header_2,\
																							servo_packet.tx.elements.id,\
																							servo_packet.tx.elements.length,\
																							servo_packet.tx.elements.cmd,\
																							servo_packet.tx.elements.args[0],\
																							servo_packet.tx.elements.args[1],\
																							servo_packet.tx.elements.args[servo_packet.tx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);

	sprintf((char *)info_buf,"RX:%X %X %X %X %X %X %X %X\r\n",servo_packet.rx.header_1,\
	                                            servo_packet.rx.header_2,\
																							servo_packet.rx.elements.id,\
																							servo_packet.rx.elements.length,\
																							servo_packet.rx.elements.cmd,\
																							servo_packet.rx.elements.args[0],\
																							servo_packet.rx.elements.args[1],\
																							servo_packet.rx.elements.args[servo_packet.rx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"Position:%hu\r\n",read_pos);
	serial_printf_string(info_buf);
	
#elif(exp == 3)
	status = servo_write_pos(1, write_pos);
	
	sprintf((char *)info_buf,"TX:%X %X %X %X %X %X %X %X %X\r\n",servo_packet.tx.header_1,\
	                                            servo_packet.tx.header_2,\
																							servo_packet.tx.elements.id,\
																							servo_packet.tx.elements.length,\
																							servo_packet.tx.elements.cmd,\
																							servo_packet.tx.elements.args[0],\
																							servo_packet.tx.elements.args[1],\
																							servo_packet.tx.elements.args[2],\
																							servo_packet.tx.elements.args[servo_packet.tx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);

	sprintf((char *)info_buf,"RX:%X %X %X %X %X %X\r\n",servo_packet.rx.header_1,\
	                                            servo_packet.rx.header_2,\
																							servo_packet.rx.elements.id,\
																							servo_packet.rx.elements.length,\
																							servo_packet.rx.elements.cmd,\
																							servo_packet.rx.elements.args[servo_packet.rx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"Target_Position:%hu\r\n",write_pos);
	serial_printf_string(info_buf);	

#elif(exp == 4)
	// 向1号舵机指定寄存器地址写入加速度、速度和位置信息（暂不执行，需调用action触发） (Write acceleration, speed, and position information to servo register address)
	status = servo_write_reg_pos_ex(1, write_acc, write_speed, write_pos); 
	
	sprintf((char *)info_buf,"TX:%X %X %X %X %X %X %X %X %X %X %X %X %X %X %X\r\n",servo_packet.tx.header_1,\
	                                            servo_packet.tx.header_2,\
																							servo_packet.tx.elements.id,\
																							servo_packet.tx.elements.length,\
																							servo_packet.tx.elements.cmd,\
																							servo_packet.tx.elements.args[0],\
																							servo_packet.tx.elements.args[1],\
																							servo_packet.tx.elements.args[2],\
																							servo_packet.tx.elements.args[3],\
																							servo_packet.tx.elements.args[4],\
																							servo_packet.tx.elements.args[5],\
																							servo_packet.tx.elements.args[6],\
																							servo_packet.tx.elements.args[7],\
																							servo_packet.tx.elements.args[8],\
																							servo_packet.tx.elements.args[servo_packet.tx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);

	sprintf((char *)info_buf,"RX:%X %X %X %X %X %X\r\n",servo_packet.rx.header_1,\
	                                            servo_packet.rx.header_2,\
																							servo_packet.rx.elements.id,\
																							servo_packet.rx.elements.length,\
																							servo_packet.rx.elements.cmd,\
																							servo_packet.rx.elements.args[servo_packet.rx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);

	// 执行1号的舵机动作指令 (Execute servo action instruction)
	status = servo_reg_action(1);               
	
	sprintf((char *)info_buf,"TX:%X %X %X %X %X %X\r\n",servo_packet.tx.header_1,\
	                                            servo_packet.tx.header_2,\
																							servo_packet.tx.elements.id,\
																							servo_packet.tx.elements.length,\
																							servo_packet.tx.elements.cmd,\
																							servo_packet.tx.elements.args[servo_packet.tx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);

	sprintf((char *)info_buf,"RX:%X %X %X %X %X %X\r\n",servo_packet.rx.header_1,\
	                                            servo_packet.rx.header_2,\
																							servo_packet.rx.elements.id,\
																							servo_packet.rx.elements.length,\
																							servo_packet.rx.elements.cmd,\
																							servo_packet.rx.elements.args[servo_packet.rx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"Target_Acc:%hu,Target_Speed:%hu,Target_Pos:%hu\r\n",write_acc, write_speed, write_pos);
	serial_printf_string(info_buf);		

#elif(exp == 5)
	// 向1号和2号舵机同时写入加速度、速度和位置信息 (Write acceleration, speed, and position information to servo 1 and 2 simultaneously)
	status = servo_sync_write_pos_ex(sync_write1, 2);      
	
	sprintf((char *)info_buf,"TX:%X %X %X %X %X %X %X %X %X %X %X %X %X %X %X %X %X %X %X %X %X\r\n",servo_packet.tx.header_1,\
	                                            servo_packet.tx.header_2,\
																							servo_packet.tx.elements.id,\
																							servo_packet.tx.elements.length,\
																							servo_packet.tx.elements.cmd,\
																							servo_packet.tx.elements.args[0],\
																							servo_packet.tx.elements.args[1],\
	
																							servo_packet.tx.elements.args[2],\
																							servo_packet.tx.elements.args[3],\
																							servo_packet.tx.elements.args[4],\
																							servo_packet.tx.elements.args[5],\
																							servo_packet.tx.elements.args[6],\
																							servo_packet.tx.elements.args[7],\
																							servo_packet.tx.elements.args[8],\
																							servo_packet.tx.elements.args[9],\
																							servo_packet.tx.elements.args[10],\

																							servo_packet.tx.elements.args[11],\
																							servo_packet.tx.elements.args[12],\
																							servo_packet.tx.elements.args[13],\
																							servo_packet.tx.elements.args[14],\
																							servo_packet.tx.elements.args[15],\
																							servo_packet.tx.elements.args[16],\
																							servo_packet.tx.elements.args[17],\
																							servo_packet.tx.elements.args[18],\
																							servo_packet.tx.elements.args[19],\
																							servo_packet.tx.elements.args[servo_packet.tx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"ServoID:%hu,Target_Acc:%hu,Target_Speed:%hu,Target_Pos:%hu\r\n",sync_write1[0][0],sync_write1[0][1],\
																																														sync_write1[0][2],sync_write1[0][3]);
	serial_printf_string(info_buf);		

	sprintf((char *)info_buf,"ServoID:%hu,Target_Acc:%hu,Target_Speed:%hu,Target_Pos:%hu\r\n",sync_write1[1][0],sync_write1[1][1],\
																																														sync_write1[1][2],sync_write1[1][3]);
	serial_printf_string(info_buf);
	
#elif(exp == 6)
	status = servo_select_mode(1, 0);
	
	sprintf((char *)info_buf,"TX:%X %X %X %X %X %X %X %X\r\n",servo_packet.tx.header_1,\
	                                            servo_packet.tx.header_2,\
																							servo_packet.tx.elements.id,\
																							servo_packet.tx.elements.length,\
																							servo_packet.tx.elements.cmd,\
																							servo_packet.tx.elements.args[0],\
																							servo_packet.tx.elements.args[1],\
																							servo_packet.tx.elements.args[servo_packet.tx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);

	sprintf((char *)info_buf,"RX:%X %X %X %X %X %X\r\n",servo_packet.rx.header_1,\
	                                            servo_packet.rx.header_2,\
																							servo_packet.rx.elements.id,\
																							servo_packet.rx.elements.length,\
																							servo_packet.rx.elements.cmd,\
																							servo_packet.rx.elements.args[servo_packet.rx.elements.length - 2]\
																							);
	serial_printf_string(info_buf);
	
	sprintf((char *)info_buf,"Running_mode:0\r\n");
	serial_printf_string(info_buf);		

#endif
	
	
//	status = servo_enable_torque(1);          // 启用1号的舵机的力矩输出 (Enable servo torque output)
//	status = servo_disable_torque(1);         // 禁用1号的舵机的力矩输出 (Disable servo torque output)
//	status = servo_cali_pos(1);               // 校准1号的当前舵机位置 (Calibrate servo position)
//	
//	status = servo_select_mode(1, 0);         // 选择1号的舵机工作模式 (Select servo working mode)
//	
//	status = servo_write_pos(1, write_pos);   // 设置1号的舵机目标位置 (Set servo target position)
//	status = servo_read_pos(1, &read_pos);    // 读取1号的舵机当前位置值 (Read servo current position value)
//	
//	status = servo_write_pos_offset(1, write_pos_offset);  // 设置1号的舵机位置偏差 (Set servo position offset)
//	status = servo_read_pos_offset(1, &read_pos_offset);   // 读取1号的舵机位置偏差 (Read servo position offset)

//	
//	status = servo_write_acc(1, write_acc);   // 设置1号的舵机加速度参数 (Set servo acceleration parameter)
//	
//	status = servo_write_speed(1, write_speed); // 设置1号的舵机目标速度 (Set servo target speed)
//	status = servo_read_speed(1, &read_speed);  // 读取1号的舵机当前速度 (Read servo current speed)
//	
//	status = servo_read_pos_speed(1, &read_pos, &read_speed); // 同时读取1号的舵机位置和速度 (Read servo position and speed)
//	
//	status = servo_write_pos_ex(1, write_acc, write_speed, write_pos); // 扩展位置控制：同时设置1号的舵机加速度、速度和位置 (Extended position control: set servo acceleration, speed, and position simultaneously)
//	status = servo_write_pwm_speed(1, write_pwm_speed);     // 设置1号的舵机PWM速度（PWM模式下生效） (Set servo PWM speed (effective in PWM mode))
//	status = servo_write_max_torque(1, write_torque);       // 设置1号的舵机最大力矩限制 (Set servo maximum torque limit)
//	status = servo_write_reg_pos_ex(1, write_acc, write_speed, write_pos); // 向1号舵机指定寄存器地址写入加速度、速度和位置信息（暂不执行，需调用action触发） (Write acceleration, speed, and position information to servo register address)
//	status = servo_reg_action(1);               // 执行1号的舵机动作指令 (Execute servo action instruction)
//	status = servo_sync_write_pos_ex(sync_write1, 2);      // 向1号和2号舵机同时写入加速度、速度和位置信息 (Write acceleration, speed, and position information to servo 1 and 2 simultaneously)
//	status = servo_sync_read_cur_pos_ex(read_id, 2, sync_read_data);  // 同部读取1号和2号舵机加速度、速度和位置信息 (Read acceleration, speed, and position information from servo 1 and 2 simultaneously)
//	
//	status = servo_read_temperture(1, &temp);   // 读取1号舵机温度值 (Read servo temperature value)
//	status = servo_read_voltage(1, &vol);       //  读取1号舵机电压值 (Read servo voltage value)
//	status = servo_read_current(1, &cur);       //  读取1号舵机电流值 (Read servo current value)
//	status = servo_read_load(1, &read_load);    //  读取1号舵机负载值 (Read servo load value)
//	status = servo_read_moving_status(1, &moving_status); //  读取1号舵机运动状态 (Read servo moving status)

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV2;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief NVIC Configuration.
  * @retval None
  */
static void MX_NVIC_Init(void)
{
  /* USART1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(USART1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(USART1_IRQn);
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
