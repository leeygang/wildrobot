#include "HX_30HM.h"
#include "usart.h"

#define MASK_SERVO(x, bit)	(((x) & ((1U) << (bit))) != 0 ? -((x) & ~((1U) << (bit))) : (x))
#define MASK_HOST(x, bit) 	((x) < 0 ? (-x) | (1U) << (bit): (x))
#define LIMIT(x, min, max) (((x) < (min)) ? (min) : ((x) > (max)) ? (max) : (x))

PacketControllerTypeDef servo_packet;

static void servo_dma_receive_event_callback(UART_HandleTypeDef *huart, uint16_t length)
{
	if(huart->Instance == USART2) {
		switch(HAL_UARTEx_GetRxEventType(huart)) {
			case HAL_UART_RXEVENT_TC:
			case HAL_UART_RXEVENT_HT:
			case HAL_UART_RXEVENT_IDLE:
				packet_write_rb(&servo_packet);
				break;

			default:
				break;
		}
		servo_packet.receive(&servo_packet);
	}
}

static void servo_uart_rx_error_callblack(UART_HandleTypeDef *huart)
{
	servo_packet.receive(&servo_packet);
}

static uint8_t write2buf(uint8_t id, uint8_t cmd, uint8_t *data, uint8_t data_len)
{
	uint8_t frame_size;

	if(id > BROADCAST_ID) {
		return 1;
	}

	switch (cmd) {
		case CMD_PING:	
		case CMD_READ:
		case CMD_WRITE:
		case CMD_REG_WRITE:
		case CMD_ACTION:
		case CMD_RESET:
		case CMD_SYNC_WRITE:
		case CMD_SYNC_READ:
			break;
		
		default:
			return 1;
	}

	frame_size = tx_frame_complete(&servo_packet, id, cmd, data, data_len);
	
	if(!lwrb_write(&servo_packet.tx_rb, (const uint8_t*)&servo_packet.tx, frame_size)) {
		return 1;
	}

	return 0;	
}

static uint8_t send_data()
{
	size_t size = lwrb_get_full(&servo_packet.tx_rb);
	uint8_t data[size];
	
	if(size > 0) {
		lwrb_read(&servo_packet.tx_rb, &data, size);
		if(servo_packet.transmit(&servo_packet, data, size)) {
			return 1;
		}	
		
		return 0;
	}

	return 1;
}
	size_t size;
static ServoStatus_t ack() 
{
	ServoStatus_t status;
	

	static uint32_t tickstart = 0;
	
	status.id = 0xFF;
	status.error_byte = 0;
	
	if(!servo_packet.rx_skip) {
		tickstart = MILLIS();

		while(1) {
			size = lwrb_get_full(&servo_packet.rx_rb);
			if(size >= servo_packet.rx_frame_length) break;
			
			if(MILLIS() - tickstart > servo_packet.rx_timeout) {
				tickstart = MILLIS();
				status.error_bits.bit_rx = 1;
				return status;
			}
		}
		
		if(unpack(&servo_packet)) {
			status.error_bits.bit_rx = 1;
		}
		else {
			status.id = servo_packet.rx.elements.id;
			status.error_byte = servo_packet.rx.elements.cmd;
		}	
	}

	return status;
}

void servo_init()
{
	packt_init(&servo_packet, &huart2, &hdma_usart2_rx, 0);
	HAL_UART_RegisterCallback(servo_packet.huart, HAL_UART_ERROR_CB_ID, servo_uart_rx_error_callblack);
	HAL_UART_RegisterRxEventCallback(&huart2, servo_dma_receive_event_callback);
	servo_packet.receive(&servo_packet);
}

ServoStatus_t servo_ping(uint8_t id)
{
	ServoStatus_t status;
	
	status.id = id;
	status.error_byte = 0;
	servo_packet.rx_skip = 0;
	servo_packet.rx_frame_length = 6;	// 6 = header + id + length + status + parameter_len + checksum
	
	if(write2buf(id, CMD_PING, NULL, 0)) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	if(send_data()) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	return ack();
}

ServoStatus_t general_write(uint8_t id, uint8_t addr, uint8_t *data, uint8_t data_len)
{
	ServoStatus_t status;
	
	uint8_t buf[1 + data_len];

#if ACK == 1
	servo_packet.rx_skip = id == 0xfe ? 1 : 0;
	servo_packet.rx_frame_length = 6;
#else 
	servo_packet.rx_skip = 1;
#endif
	
	status.id = id;
	status.error_byte = 0;
	buf[0] = addr;
	
	for(uint8_t i = 0; i < data_len; i++) {
		buf[1 + i] = data[i];
	}
	
	if(write2buf(id, CMD_WRITE, buf, sizeof(buf))) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	if(send_data()) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	
	return ack();
}

ServoStatus_t general_read(uint8_t id, uint8_t addr, uint8_t *data, uint8_t data_len)
{
	ServoStatus_t status;
	uint8_t buf[2];

	buf[0] = addr;
	buf[1] = data_len;
	servo_packet.rx_skip = 0;
	servo_packet.rx_frame_length = 6 + data_len;	// 6 = header + id + length + status + parameter_len + checksum

	if(write2buf(id, CMD_READ, buf, sizeof(buf))) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	if(send_data()) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	status = ack();

	if(status.error_bits.bit_rx) {
		return status;
	}
	
	for (uint8_t i = 0; i < data_len; i++)
	{
		data[i] = servo_packet.rx.elements.args[i];
	}
	
	return status;
}

ServoStatus_t reg_write(uint8_t id, uint8_t addr, uint8_t *data, uint8_t data_len)
{
	ServoStatus_t status;
	
	uint8_t buf[1 + data_len];
	
#if ACK == 1
	servo_packet.rx_skip = id == 0xfe ? 1 : 0;
	servo_packet.rx_frame_length = 6;
#else 
	servo_packet.rx_skip = 1;
#endif
	
	status.id = id;
	status.error_byte = 0;
	buf[0] = addr;

	for(uint8_t i = 0; i < data_len; i++) {
		buf[1 + i] = data[i];
	}

	if(write2buf(id, CMD_REG_WRITE, buf, sizeof(buf))) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	if(send_data()) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	return ack();
}

ServoStatus_t reg_action(uint8_t id)
{
	ServoStatus_t status;
	
#if ACK == 1
	servo_packet.rx_skip = id == 0xfe ? 1 : 0;
	servo_packet.rx_frame_length = 6;
#else 
	servo_packet.rx_skip = 1;
#endif
	
	status.id = id;
	status.error_byte = 0;
	
	if(write2buf(id, CMD_ACTION, NULL, 0)) {
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	if(send_data()) {
		status.error_bits.bit_tx = 1;
		return status;
	}

	return ack();
}
	
ServoStatus_t sync_write(uint8_t addr, uint8_t *data, uint8_t data_len, uint8_t parameter_len)
{
	ServoStatus_t status;

	uint8_t frame_size;
	uint8_t total_size = data_len + 2;
	uint8_t buf[total_size];
	
	buf[0] = addr;
	buf[1] = parameter_len;

	status.id = BROADCAST_ID;
	status.error_byte = 0;
	
	for(uint8_t i = 0; i < data_len; i++) {
		buf[2 + i] = data[i];
	}
	
	if(write2buf(BROADCAST_ID, CMD_SYNC_WRITE, buf, sizeof(buf))) {
		status.error_bits.bit_tx = 1;
	}	

	if(send_data()) {
		status.error_bits.bit_tx = 1;
	}
	
	return status;
} 

ServoStatus_t sync_read(uint8_t addr, uint8_t byte_num, uint8_t *id, uint8_t id_num, uint8_t (*data)[byte_num])
{
	ServoStatus_t status;
	uint8_t buf[1 + byte_num + id_num];
	buf[0] = addr;
	buf[1] = byte_num;
	status.id = BROADCAST_ID;
	status.error_byte = 0;
	servo_packet.rx_skip = 0;
	servo_packet.rx_frame_length = 6 + byte_num;

	for(uint8_t i = 0; i < id_num; i++) {
		buf[2 + i] = id[i];
	}

	if(write2buf(BROADCAST_ID, CMD_SYNC_READ, buf, sizeof(buf))) {
		status.error_bits.bit_tx = 1;
		return status;
	}	

	if(send_data()) {
		status.error_bits.bit_tx = 1;
		return status;
	}

	for(uint8_t i = 0; i < id_num; i++) {
		status = ack();
		if(status.error_bits.bit_rx) {
			return status;
		}
		
		for(uint8_t j = 0; j < byte_num; j++) {
			data[i][j] = servo_packet.rx.elements.args[j];
		}
	}

	return status;
}

ServoStatus_t servo_enable_torque(uint8_t id)
{
	uint8_t data = 1;

	return general_write(id, REG_TORQUE_ENABLE, &data, 1);
}

ServoStatus_t servo_disable_torque(uint8_t id)
{
	uint8_t data = 0;

	return general_write(id, REG_TORQUE_ENABLE, &data, 1);
}

ServoStatus_t servo_cali_pos(uint8_t id)
{
	uint8_t data = 128;

	return general_write(id, REG_TORQUE_ENABLE, &data, 1);
}

ServoStatus_t servo_select_mode(uint8_t id, uint8_t mode)
{
	uint8_t data;
	
	data = mode;
	
	return general_write(id, REG_MODE, &data, 1);
}

ServoStatus_t servo_write_pos_offset(uint8_t id, int16_t offset)
{
	uint16_t data;

	offset = LIMIT(offset, -2047, 2047);
	data = (uint16_t)MASK_HOST(offset, 11);

	return general_write(id, REG_POS_OFFSET_L, (uint8_t *)&data, sizeof(data));
}

ServoStatus_t servo_write_pos(uint8_t id, int16_t pos)
{
	uint16_t data;
	
	pos = LIMIT(pos, -30719, 30719);
	data = (uint16_t)MASK_HOST(pos, 15);

	return general_write(id, REG_GOAL_POSITION_L, (uint8_t *)&data, sizeof(data));
}

ServoStatus_t servo_write_acc(uint8_t id, uint8_t acc)
{
	uint8_t data;
	
	data = LIMIT(acc, 0, 254);

	return general_write(id, REG_ACC, &data, sizeof(data));
}

ServoStatus_t servo_write_speed(uint8_t id, int16_t speed)
{
	uint16_t data;
	
	speed = LIMIT(speed, -3400, 3400);
	data = (uint16_t)MASK_HOST(speed, 15);
	
	return general_write(id, REG_GOAL_SPEED_L, (uint8_t *)&data, sizeof(data));
}

ServoStatus_t servo_write_pos_ex(uint8_t id, uint8_t acc, int16_t speed, int16_t pos)
{
	uint8_t data[7];
	uint8_t	_acc;
	uint16_t _pos;
	uint16_t _speed;
	
	acc = LIMIT(acc, 0, 254);
	speed = LIMIT(speed, -3400, 3400);
	pos = LIMIT(pos, -30719, 30719);
	
	_pos = (uint16_t)MASK_HOST(pos, 15);

	
	data[0] = acc;
	word2bytes(&servo_packet, _pos, &data[1], &data[2]);
	data[3] = 0;
	data[4] = 0;
	word2bytes(&servo_packet, _pos, &data[5], &data[6]);

	return general_write(id, REG_ACC, (uint8_t *)&data, sizeof(data));
}

ServoStatus_t servo_write_pwm_speed(uint8_t id, int16_t speed)
{
	uint16_t data;
	
	speed = LIMIT(speed, -1000, 1000);
	data = (uint16_t)MASK_HOST(speed, 10);
	
	return general_write(id, REG_PWM_SPEED_L, (uint8_t *)&data, sizeof(data));
}

ServoStatus_t servo_write_max_torque(uint8_t id, uint16_t torque)
{
	uint16_t data;
	
	data = LIMIT(torque, 0, 1000);
	
	return general_write(id, REG_MAX_TORQUE_L, (uint8_t *)&data, sizeof(data));
}

ServoStatus_t servo_write_reg_pos_ex(uint8_t id, uint8_t acc, int16_t speed, int16_t pos)
{
	uint8_t data[7];
	uint8_t	_acc;
	uint16_t _pos;
	uint16_t _speed;
	
	acc = LIMIT(acc, 0, 254);
	speed = LIMIT(speed, -3400, 3400);
	pos = LIMIT(pos, -30719, 30719);
	
	_pos = (uint16_t)MASK_HOST(pos, 15);
	_speed = (uint16_t)MASK_HOST(speed, 15);
	
	data[0] = acc;
	word2bytes(&servo_packet, _pos, &data[1], &data[2]);
	data[3] = 0;
	data[4] = 0;
	word2bytes(&servo_packet, _pos, &data[5], &data[6]);

	return reg_write(id, REG_ACC, (uint8_t *)&data, sizeof(data));
}

ServoStatus_t servo_reg_action(uint8_t id)
{
	return reg_action(id);
}

ServoStatus_t servo_sync_write_pos_ex(int16_t (*data)[4], uint8_t id_num)
{
	ServoStatus_t status;

	uint8_t id;
	uint8_t acc;
	int16_t speed;
	int16_t pos;
	uint16_t u16_speed;
	uint16_t u16_pos;
	uint8_t buf[id_num * 8];

	if(id_num > 30) {
		status.id = 0xFF;
		status.error_byte = 0;
		status.error_bits.bit_tx = 1;
		return status;
	}
	
	for (uint8_t i = 0; i < id_num; i++)
	{
		id = (uint8_t)data[i][0];
		acc = (uint8_t)data[i][1];
		speed = (int16_t)data[i][2];
		pos = (int16_t)data[i][3];
		
		acc = LIMIT(acc, 0, 254);
		speed = LIMIT(speed, -3400, 3400);
		pos = LIMIT(pos, -30719, 30719);

		u16_pos = (uint16_t)MASK_HOST(pos, 15);
		u16_speed = (uint16_t)MASK_HOST(speed, 15);

		buf[i * 8] = id;
		buf[(i * 8) + 1] = acc;
		word2bytes(&servo_packet, u16_pos, &buf[(i * 8) + 2], &buf[(i * 8) + 3]);
		buf[(i * 8) + 4] = 0;
		buf[(i * 8) + 5] = 0;
		word2bytes(&servo_packet, u16_speed, &buf[(i * 8) + 6], &buf[(i * 8) + 7]);
	}
	
	return sync_write(REG_ACC, buf, sizeof(buf), 7);
}

ServoStatus_t servo_sync_read_cur_pos_ex(uint8_t *id, uint8_t id_num, int16_t (*data)[5])
{
	ServoStatus_t status;
	uint8_t byte_len = 8;
	uint8_t read_data[id_num][byte_len];
	uint16_t u16_pos[id_num];
	uint16_t u16_speed[id_num];
	uint16_t u16_load[id_num];
	
	status = sync_read(REG_PRESENT_POSITION_L, byte_len, id, id_num, read_data);
	
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		return status;
	}
	
	
	for(uint8_t i = 0; i < id_num; i++) {
		u16_pos[i] = bytes2word(&servo_packet, &read_data[i][0], &read_data[i][1]);
		u16_speed[i] = bytes2word(&servo_packet, &read_data[i][2], &read_data[i][3]);
		u16_load[i] = bytes2word(&servo_packet, &read_data[i][4], &read_data[i][5]);
		
		
		data[i][0] = (int16_t)MASK_SERVO(u16_pos[i], 15);
		data[i][1] = (int16_t)MASK_SERVO(u16_speed[i], 15);
		data[i][2] = (int16_t)MASK_SERVO(u16_load[i], 10);
		data[i][3] = (int16_t)read_data[i][6];
		data[i][4] = (int16_t)read_data[i][7];
	}
	
	return status;
}
	
ServoStatus_t servo_read_pos_offset(uint8_t id, int16_t *offset)
{
	ServoStatus_t status;
	uint8_t data[2];
	uint16_t _offset;

	status = general_read(id, REG_POS_OFFSET_L, data, 2);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*offset = NULL;
		return status;
	}

	_offset = bytes2word(&servo_packet, data, data + 1);
	*offset = (int16_t)MASK_SERVO(_offset, 11);
		
	return  status;
}

ServoStatus_t servo_read_pos(uint8_t id, int16_t *pos)
{
	ServoStatus_t status;
	uint8_t data[2];
	uint16_t _pos;

	status = general_read(id, REG_PRESENT_POSITION_L, data, 2);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*pos = NULL;
		return status;
	}

	_pos = bytes2word(&servo_packet, data, data + 1);
	*pos = (int16_t)MASK_SERVO(_pos, 15);
	
	return  status;
}

ServoStatus_t servo_read_speed(uint8_t id, int16_t *speed)
{
	ServoStatus_t status;	
	uint8_t data[2];
	uint16_t _speed;

	status = general_read(id, REG_PRESENT_SPEED_L, data, 2);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*speed = NULL;
		return status;
	}

	_speed = bytes2word(&servo_packet, data, data + 1);
	*speed = (int16_t)MASK_SERVO(_speed, 15);
		
	return  status;
}

ServoStatus_t servo_read_pos_speed(uint8_t id, int16_t *pos, int16_t *speed)
{

	ServoStatus_t status;	
	uint8_t data[4];
	uint16_t _pos;
	uint16_t _speed;

	status = general_read(id, REG_PRESENT_POSITION_L, data, 4);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*pos = 0;
		*speed = 0;
		return status;
	}

	_pos = bytes2word(&servo_packet, data, data + 1);
	*pos = (int16_t)MASK_SERVO(_pos, 15);
		
	_speed = bytes2word(&servo_packet, data + 2, data + 3);
	*speed = (int16_t)MASK_SERVO(_speed, 15);

	return  status;
}

ServoStatus_t servo_read_temperture(uint8_t id, uint8_t *temp)
{
	ServoStatus_t status;	
	uint8_t data[1];

	status = general_read(id, REG_PRESENT_TEMPERATURE, data, 1);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*temp = NULL;
		return status;
	}

	*temp = data[0];
		
	return  status;
}

ServoStatus_t servo_read_voltage(uint8_t id, uint8_t *vol)
{
	ServoStatus_t status;	
	uint8_t data[1];

	status = general_read(id, REG_PRESENT_VOLTAGE, data, 1);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*vol = NULL;
		return status;
	}

	*vol = data[0];
		
	return  status;
}

ServoStatus_t servo_read_current(uint8_t id, uint16_t *cur)
{
	ServoStatus_t status;	
	uint8_t data[2];
	uint16_t _cur;

	status = general_read(id, REG_PRESENT_CURRENT_L, data, 2);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*cur = NULL;
		return status;
	}

	_cur = bytes2word(&servo_packet, data, data + 1);
	*cur = _cur;
		
	return  status;
}

ServoStatus_t servo_read_load(uint8_t id, int16_t *load)
{
	ServoStatus_t status;	
	uint8_t data[2];
	int16_t _load;

	status = general_read(id, REG_PRESENT_LOAD_L, data, 2);
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*load = NULL;
		return status;
	}

	_load = bytes2word(&servo_packet, data, data + 1);
	*load = (int16_t)MASK_SERVO(_load, 10);
		
	return  status;
}

ServoStatus_t servo_read_moving_status(uint8_t id, uint8_t *moving_status)
{
	ServoStatus_t status;	
	uint8_t data;

	status = general_read(id, REG_MOVING_STATUS, &data, sizeof(data));
	if(status.error_bits.bit_tx || status.error_bits.bit_rx) {
		*moving_status = NULL;
		return status;
	}

	*moving_status = data;
		
	return  status;
}

