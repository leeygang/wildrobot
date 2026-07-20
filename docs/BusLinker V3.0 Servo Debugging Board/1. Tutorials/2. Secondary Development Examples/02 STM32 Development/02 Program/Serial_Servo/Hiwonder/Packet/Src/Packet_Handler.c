#include "Packet_Handler.h"
#include "string.h"
	
uint8_t tx_buf[255];
uint8_t rx_buf[255];

uint8_t packet_rx_dma_buf[255];

static uint8_t tramsmit_data(struct PacketController *self, const uint8_t *pdata, uint16_t size)
{
    return HAL_UART_Transmit(self->huart, pdata, size, self->tx_timeout);
}

static uint8_t receive_dma_data(struct PacketController *self)
{
	return HAL_UARTEx_ReceiveToIdle_DMA(self->huart, self->rx_dma_buf, self->rx_dma_buf_size);
}

void packt_init(PacketControllerTypeDef *self, UART_HandleTypeDef *huart, DMA_HandleTypeDef *hdma_rx, uint8_t end) 
{
    memset(self, 0, sizeof(PacketControllerTypeDef));
    self->endianness = end;
    self->huart = huart;
	self->hdma_rx = hdma_rx;
    self->transmit = tramsmit_data;
    self->receive = receive_dma_data;
	self->rx_dma_buf = packet_rx_dma_buf;
	self->rx_dma_buf_size = sizeof(packet_rx_dma_buf);
    self->tx_timeout = 10;
	self->rx_timeout = 10;
	lwrb_init(&self->tx_rb, tx_buf, sizeof(tx_buf));
	lwrb_init(&self->rx_rb, rx_buf, sizeof(rx_buf));
}

void word2bytes(PacketControllerTypeDef *self, uint16_t word, uint8_t *bytes_l, uint8_t *bytes_h)
{
    if (self->endianness) {
	    *bytes_l = (word >> 8);
		*bytes_h = (word & 0xff);
    }
    else {
		*bytes_h = (word >> 8);
		*bytes_l = (word & 0xff);
    }
}

uint16_t bytes2word(PacketControllerTypeDef *self, uint8_t *bytes_l, uint8_t *bytes_h)
{
    uint16_t word;

    if (self->endianness) {
		word = *bytes_l;
		word <<= 8;
		word |= *bytes_h;
    }
    else {
		word = *bytes_h;
		word <<= 8;
		word |= *bytes_l;
    }

    return word;
}

void set_tx_timeout(PacketControllerTypeDef *self, uint32_t timeout)
{
    self->tx_timeout = timeout;
}

void set_rx_timeout(PacketControllerTypeDef *self, uint32_t timeout)
{
    self->rx_timeout = timeout;
}

uint8_t data_check(const uint8_t buf[], uint8_t len)
{
    uint16_t temp = 0;
    for (int i = 2; i < len; ++i) {
        temp += buf[i];
    }
    return (uint8_t)(~temp);
}

uint8_t tx_frame_complete(PacketControllerTypeDef *self, uint8_t id, uint8_t cmd, uint8_t *data, uint8_t data_len)
{
	uint8_t frame_len;
	self->tx.header_1 = FRAME_HEADER_1;
	self->tx.header_2 = FRAME_HEADER_2;
	self->tx.elements.id = id;
	self->tx.elements.length = 2 + data_len;
	self->tx.elements.cmd = cmd;
	frame_len = 4 + self->tx.elements.length;
	
	for(uint8_t i = 0; i < data_len; i++) {
		self->tx.elements.args[i] = data[i];
	}
	
	self->tx.elements.args[data_len] = data_check((const uint8_t*)&self->tx, frame_len - 1);
	
	return frame_len;
}

void packet_write_rb(PacketControllerTypeDef *self)
{
	static size_t old_pos;
	size_t pos;
	
	pos = self->rx_dma_buf_size -__HAL_DMA_GET_COUNTER(self->hdma_rx); 
	
	if(pos != old_pos) {
		if(!self->rx_skip) {
			if(pos > old_pos) {
				lwrb_write(&self->rx_rb, &self->rx_dma_buf[old_pos], pos - old_pos);
			}
			else {
				lwrb_write(&self->rx_rb, &self->rx_dma_buf[old_pos], self->rx_dma_buf_size - old_pos);
				if(pos > 0) {
					lwrb_write(&self->rx_rb, &self->rx_dma_buf[0], pos);
				}
			}		
		}
		old_pos = pos;
	}
}


uint8_t unpack(PacketControllerTypeDef *self)
{
	uint8_t error;
	uint8_t check_value;
	size_t available;
	available = lwrb_get_full(&self->rx_rb);
	self->rx_status = PACKET_HEADER_1;

	while(available) {
		switch(self->rx_status) {
			case PACKET_HEADER_1:
				lwrb_read(&self->rx_rb, &self->rx.header_1, 1);
				self->rx_status = self->rx.header_1 == FRAME_HEADER_1 ? PACKET_HEADER_2: PACKET_HEADER_1;
				break;
			
			case PACKET_HEADER_2:
				lwrb_read(&self->rx_rb, &self->rx.header_2, 1);
				self->rx_status = self->rx.header_2 == FRAME_HEADER_2 ? PACKET_ID: PACKET_HEADER_1;				
				break;
			
			case PACKET_ID:
				lwrb_read(&self->rx_rb, &self->rx.elements.id, 1);
				self->rx_status = self->rx.elements.id <= BROADCAST_ID ? PACKET_DATA_LENGTH: PACKET_HEADER_1;	
				break;
			
			case PACKET_DATA_LENGTH:
				lwrb_read(&self->rx_rb, &self->rx.elements.length, 1);
				self->rx_status = self->rx.elements.length <= MAX_FRAME_SIZE + 1  ? PACKET_CMD: PACKET_HEADER_1;
				break;

			case PACKET_CMD:
				lwrb_read(&self->rx_rb, &self->rx.elements.cmd, 1);
				self->rx_status = self->rx.elements.cmd <= CMD_SYNC_READ ? PACKET_PARAMETERS: PACKET_HEADER_1;
				break;
			
			case PACKET_PARAMETERS:
				lwrb_read(&self->rx_rb, &self->rx.elements.args, self->rx.elements.length - 2);
				self->rx_status = PACKET_CHECKSUM;
				break;
			
			case PACKET_CHECKSUM:
				lwrb_read(&self->rx_rb, &self->rx.elements.args[self->rx.elements.length - 2], 1);
				check_value = data_check((const uint8_t*)&self->rx, self->rx.elements.length + 3);
				self->rx_status = self->rx.elements.args[self->rx.elements.length - 2] == check_value ? PACKET_FINISH : PACKET_HEADER_1;
				break;
			
			default:
				break;
		}
		
		if(self->rx_status == PACKET_FINISH) {
			break;
		}
		
		available = lwrb_get_full(&self->rx_rb);
	}

	return error = self->rx_status == PACKET_FINISH ? 0 : 1;
}