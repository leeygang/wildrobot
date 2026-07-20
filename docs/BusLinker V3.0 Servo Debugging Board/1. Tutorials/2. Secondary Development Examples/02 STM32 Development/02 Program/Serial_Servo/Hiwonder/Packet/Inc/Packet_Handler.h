/**
 * @file Packet_Handler.h
 * @author Min
 * @brief Packet processing porting layer header file
 * @version 0.1
 * @date 2025-10-18
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PACKET_HANDLER_H
#define PACKET_HANDLER_H

#include "stdint.h"
#include "stm32f1xx_hal.h"
#include "HX_30HM_Def.h"
#include "lwrb.h"


#define MAX_FRAME_SIZE 250
#define MILLIS()	HAL_GetTick()
	
/**
 * @enum PacketStatus
 * @brief 数据包解析状态枚举 (Packet parsing status enumeration)
 */
typedef enum
{
	PACKET_HEADER_1 = 0,  /**< 第一个帧头字节 (First header byte) */
	PACKET_HEADER_2,      /**< 第二个帧头字节 (Second header byte) */
	PACKET_ID,            /**< ID字段 (ID field) */
	PACKET_DATA_LENGTH,   /**< 数据长度字段 (Data length field) */
	PACKET_CMD,           /**< 命令字段 (Command field) */
	PACKET_PARAMETERS,    /**< 参数字段 (Parameter field) */
	PACKET_CHECKSUM,      /**< 校验和字段 (Checksum field) */
	PACKET_FINISH         /**< 数据包解析完成 (Packet completed) */
}PacketStatus;


/**
 * @struct PacketTypeDef
 * @brief 数据包结构定义 (Packet structure definition)
 * @details 定义了舵机通信数据包的格式，包含帧头、ID、长度、命令和参数 (Defines servo communication packet format)
 */
#pragma pack(1)
typedef struct {
    uint8_t header_1;     /**< 第一个帧头字节 (First header byte) */
    uint8_t header_2;     /**< 第二个帧头字节 (Second header byte) */

    union {
        struct {
		uint8_t id;                             /**< 舵机ID (Servo ID) */
            uint8_t length;                     /**< 数据包长度 (Packet length) */
            uint8_t cmd;                        /**< 命令字节 (Command byte) */
            uint8_t args[MAX_FRAME_SIZE];       /**< 参数数据 (Parameter data) */
        } elements;                             /**< 分解的数据包元素 (Decomposed packet elements) */
        uint8_t data_raw[MAX_FRAME_SIZE + 3];   /**< 发送/接收参数数组 (Transmit/Receive parameter buffer) */
    };
} PacketTypeDef;
#pragma pack()

/**
 * @struct PacketControllerTypeDef
 * @brief 数据包控制器结构 (Packet controller structure)
 * @details 管理数据包的发送、接收和解析过程 (Manages packet transmission, reception and parsing)
 */
typedef struct PacketController {
    UART_HandleTypeDef *huart;       /**< UART句柄 (UART handle) */
	DMA_HandleTypeDef *hdma_rx;      /**< DMA接收句柄 (DMA receive handle) */
	
    uint8_t endianness;              /**< 字节序设置 (Endianness setting) 0-小端, 1-大端 */
	
    uint8_t tx_ready;                /**< 发送就绪标志 (Transmit ready flag) */
    uint8_t tx_index;                /**< 发送索引 (Transmit index) */
	uint32_t tx_timeout;             /**< 发送超时时间 (Transmit timeout) */
	
    uint8_t rx_index;                /**< 接收索引 (Receive index) */
    uint8_t rx_count;                /**< 接收计数 (Receive count) */
	uint8_t rx_skip;                 /**< 接收跳过标志 (Receive skip flag) */
    uint8_t rx_frame_length;         /**< 接收帧长度 (Receive frame length) */
	uint8_t *rx_dma_buf;             /**< DMA接收缓冲区 (DMA receive buffer) */
	uint8_t rx_dma_buf_size;         /**< DMA接收缓冲区大小 (DMA receive buffer size) */
	uint32_t rx_timeout;             /**< 接收超时时间 (Receive timeout) */
	
	lwrb_t tx_rb;                    /**< 发送环形缓冲区 (Transmit ring buffer) */
	lwrb_t rx_rb;                    /**< 接收环形缓冲区 (Receive ring buffer) */
    
    PacketTypeDef tx;                /**< 发送数据包 (Transmit packet) */
    PacketTypeDef rx;                /**< 接收数据包 (Receive packet) */
	PacketStatus rx_status;          /**< 接收状态 (Receive status) */

	/**
     * @brief 发送函数指针 (Transmit function pointer)
     * @param self 控制器指针 (Controller pointer)
     * @param pdata 数据指针 (Data pointer)
     * @param size 数据大小 (Data size)
     * @return 发送结果 (Transmit result)
     */
    uint8_t (*transmit)(struct PacketController *self, const uint8_t *pdata, uint16_t size);
    
    /**
     * @brief 接收函数指针 (Receive function pointer)
     * @param self 控制器指针 (Controller pointer)
     * @return 接收结果 (Receive result)
     */
    uint8_t (*receive)(struct PacketController *self);

} PacketControllerTypeDef;

/**
 * @brief 初始化数据包控制器 (Initialize packet controller)
 * @param self 控制器指针 (Controller pointer)
 * @param huart UART句柄 (UART handle)
 * @param hdma_rx DMA接收句柄 (DMA receive handle)
 * @param end 字节序设置 (Endianness setting)
 */
void packt_init(PacketControllerTypeDef *self, UART_HandleTypeDef *huart, DMA_HandleTypeDef *hdma_rx, uint8_t end);

/**
 * @brief 设置发送超时时间 (Set transmit timeout)
 * @param self 控制器指针 (Controller pointer)
 * @param timeout 超时时间 (Timeout value)
 */
void set_tx_timeout(PacketControllerTypeDef *self, uint32_t timeout);

/**
 * @brief 设置接收超时时间 (Set receive timeout)
 * @param self 控制器指针 (Controller pointer)
 * @param timeout 超时时间 (Timeout value)
 */
void set_rx_timeout(PacketControllerTypeDef *self, uint32_t timeout);

/**
 * @brief 数据校验 (Data check)
 * @param buf 数据缓冲区 (Data buffer)
 * @param len 数据长度 (Data length)
 * @return 校验结果 (Check result)
 */
uint8_t data_check(const uint8_t buf[], uint8_t len);

/**
 * @brief 将字转换为字节 (Convert word to bytes)
 * @param self 控制器指针 (Controller pointer)
 * @param word 字数据 (Word data)
 * @param bytes_l 低位字节指针 (Low byte pointer)
 * @param bytes_h 高位字节指针 (High byte pointer)
 */
void word2bytes(PacketControllerTypeDef *self, uint16_t word, uint8_t *bytes_l, uint8_t *bytes_h);

/**
 * @brief 将字节转换为字 (Convert bytes to word)
 * @param self 控制器指针 (Controller pointer)
 * @param bytes_l 低位字节指针 (Low byte pointer)
 * @param bytes_h 高位字节指针 (High byte pointer)
 * @return 转换后的字数据 (Converted word data)
 */
uint16_t bytes2word(PacketControllerTypeDef *self, uint8_t *bytes_l, uint8_t *bytes_h);

/**
 * @brief 填充发送帧 (Fill transmit frame)
 * @param self 控制器指针 (Controller pointer)
 * @param id 舵机ID (Servo ID)
 * @param cmd 命令字节 (Command byte)
 * @param data 数据指针 (Data pointer)
 * @param data_len 数据长度 (Data length)
 * @return 构建结果 (Construction result)
 */
uint8_t tx_frame_complete(PacketControllerTypeDef *self, uint8_t id, uint8_t cmd, uint8_t *data, uint8_t data_len);

/**
 * @brief 将数据包写入环形缓冲区 (Write packet to ring buffer)
 * @param self 控制器指针 (Controller pointer)
 */
void packet_write_rb(PacketControllerTypeDef *self);

/**
 * @brief 解析数据包 (Unpack packet)
 * @param self 控制器指针 (Controller pointer)
 * @return 解析结果 (Unpack result)
 */
uint8_t unpack(PacketControllerTypeDef *self);

#endif
