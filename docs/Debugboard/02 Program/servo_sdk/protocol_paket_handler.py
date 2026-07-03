
TXPACKET_MAX_LEN = 250
RXPACKET_MAX_LEN = 250

# for Protocol Packet
PKT_HEADER0 = 0
PKT_HEADER1 = 1
PKT_ID = 2
PKT_LENGTH = 3
PKT_INSTRUCTION = 4
PKT_ERROR = 4
PKT_PARAMETER0 = 5

# Protocol Error bit
ERRBIT_VOLTAGE = 1      #0x01
ERRBIT_SENSOR = 2       #0x02
ERRBIT_OVERHEAT = 4     #0x04
ERRBIT_CURRENT = 8      #0x08
ERRBIT_ANGLE = 16       #0x10
ERRBIT_OVERLOAD = 32    #0x20

BROADCAST_ID = 254      #0xFE

# Instruction for Servo Protocol
INST_PING = 1           #0x01
INST_READ = 2           #0x02
INST_WRITE = 3          #0x03
INST_REG_WRITE = 4      #0x04
INST_ACTION = 5         #0x05
INST_RESET = 6          #0x06
INST_SYNC_READ = 130    #0x82
INST_SYNC_WRITE = 131   #0x83


# Communication Result
COMM_SUCCESS = 0        # tx or rx packet communication success
COMM_PORT_BUSY = -1     # Port is busy (in use)
COMM_TX_FAIL = -2       # Failed transmit instruction packet
COMM_RX_FAIL = -3       # Failed get status packet
COMM_TX_ERROR = -4      # Incorrect instruction packet
COMM_RX_WAITING = -5    # Now receiving status packet
COMM_RX_TIMEOUT = -6    # There is no status packet
COMM_RX_CORRUPT = -7    # Incorrect status packet
COMM_NOT_AVAILABLE = -9 #Protocol does not support this function

class PacketHandler(object):
    def __init__(self, PortHandler, endianness):
        self.PortHandler = PortHandler
        self.endianness = endianness # 0: little-endian   1:big-endian

    def getEndian(self):
        return self.endianness

    def setEndian(self, e):
        self.endianness = e

    def toHost(self, a, b):
        if (a & (1 << b)):
            return -(a & ~(1 << b))
        else:
            return a

    def toServo(self, a, b):
        if (a < 0):
            return (-a | (1 << b))
        else:
            return a

    def makeWord16(self, a, b):
        if self.endianness == 0:
            return (a & 0xFF) | ((b & 0xFF) << 8)
        else:
            return (b & 0xFF) | ((a & 0xFF) << 8)

    def makeWord32(self, a, b):
        return (a & 0xFFFF) | (b & 0xFFFF) << 16

    def getLowWord32(self, l):
        return l & 0xFFFF

    def get_Highword32(self, h):
        return (h >> 16) & 0xFFFF

    def getLowByte(self, w):
        if self.endianness == 0:
            return w & 0xFF
        else:
            return (w >> 8) & 0xFF

    def getHighByte(self, w):
        if self.endianness == 0:
            return (w >> 8) & 0xFF
        else:
            return w & 0xFF

    def getTxRxResult(self, result):
        if result == COMM_SUCCESS:
            return "[TxRxResult] Communication success!"
        elif result == COMM_PORT_BUSY:
            return "[TxRxResult] Port is in use!"
        elif result == COMM_TX_FAIL:
            return "[TxRxResult] Failed transmit instruction packet!"
        elif result == COMM_RX_FAIL:
            return "[TxRxResult] Failed get status packet from device!"
        elif result == COMM_TX_ERROR:
            return "[TxRxResult] Incorrect instruction packet!"
        elif result == COMM_RX_WAITING:
            return "[TxRxResult] Now receiving status packet!"
        elif result == COMM_RX_TIMEOUT:
            return "[TxRxResult] There is no status packet!"
        elif result == COMM_RX_CORRUPT:
            return "[TxRxResult] Incorrect status packet!"
        elif result == COMM_NOT_AVAILABLE:
            return "[TxRxResult] Protocol does not support this function!"
        else:
            return ""

    def getRxPacketError(self, error):
        if error & ERRBIT_VOLTAGE:
            return "[ServoStatus] Input voltage error!"

        if error & ERRBIT_SENSOR:
            return "[ServoStatus] Sensor error!"

        if error & ERRBIT_OVERHEAT:
            return "[ServoStatus] Overheat error!"

        if error & ERRBIT_CURRENT:
            return "[ServoStatus] Current error!"

        if error & ERRBIT_ANGLE:
            return "[ServoStatus] Angle error!"

        if error & ERRBIT_OVERLOAD:
            return "[ServoStatus] Overload error!"

        return ""

    def txPacket(self, txpacket):
        checksum = 0
        total_packet_length = txpacket[PKT_LENGTH] + 4  # 4: HEADER0 HEADER1 ID LENGTH

        if self.PortHandler.is_using:
            return COMM_PORT_BUSY
        self.PortHandler.is_using = True

        # check max packet length
        if total_packet_length > TXPACKET_MAX_LEN:
            self.PortHandler.is_using = False
            return COMM_TX_ERROR

        # make packet header
        txpacket[PKT_HEADER0] = 0xFF
        txpacket[PKT_HEADER1] = 0xFF

        # add a checksum to the packet
        for index in range(2, total_packet_length - 1):  # except header, checksum
            checksum += txpacket[index]

        txpacket[total_packet_length - 1] = ~checksum & 0xFF

        # tx packet
        self.PortHandler.clearPort()
        written_packet_length = self.PortHandler.writePort(txpacket)
        if total_packet_length != written_packet_length:
            self.PortHandler.is_using = False
            return COMM_TX_FAIL

        return COMM_SUCCESS

    def rxPacket(self):
        rxpacket = []

        result = COMM_TX_FAIL
        checksum = 0
        rx_length = 0
        wait_length = 6  # minimum length (HEADER0 HEADER1 ID LENGTH ERROR CHKSUM)

        while True:
            rxpacket.extend(self.PortHandler.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                # find packet header
                index = 0
                for index in range(0, (rx_length - 1)):
                    if (rxpacket[index] == 0xFF) and (rxpacket[index + 1] == 0xFF):
                        break

                if index == 0:  # found at the beginning of the packet
                    if (rxpacket[PKT_ID] > 0xFD) or (rxpacket[PKT_LENGTH] > RXPACKET_MAX_LEN) or (
                            rxpacket[PKT_ERROR] > 0x7F):
                        # unavailable ID or unavailable Length or unavailable Error
                        # remove the first byte in the packet
                        del rxpacket[0]
                        rx_length -= 1
                        continue

                    # re-calculate the exact length of the rx packet
                    if wait_length != (rxpacket[PKT_LENGTH] + PKT_LENGTH + 1):
                        wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1
                        continue

                    if rx_length < wait_length:
                        # check timeout
                        if self.PortHandler.isPacketTimeout():
                            if rx_length == 0:
                                result = COMM_RX_TIMEOUT
                            else:
                                result = COMM_RX_CORRUPT
                            break
                        else:
                            continue

                    # calculate checksum
                    for i in range(2, wait_length - 1):  # except header, checksum
                        checksum += rxpacket[i]
                    checksum = ~checksum & 0xFF

                    # verify checksum
                    if rxpacket[wait_length - 1] == checksum:
                        result = COMM_SUCCESS
                    else:
                        result = COMM_RX_CORRUPT
                    break

                else:
                    # remove unnecessary packets
                    del rxpacket[0: index]
                    rx_length -= index

            else:
                # check timeout
                if self.PortHandler.isPacketTimeout():
                    if rx_length == 0:
                        result = COMM_RX_TIMEOUT
                    else:
                        result = COMM_RX_CORRUPT
                    break

        self.PortHandler.is_using = False
        return rxpacket, result

    def txRxPacket(self, txpacket):
        rxpacket = None
        error = 0

        # tx packet
        result = self.txPacket(txpacket)
        if result != COMM_SUCCESS:
            return rxpacket, result, error

        # (ID == Broadcast ID) == no need to wait for status packet or not available
        if txpacket[PKT_ID] == BROADCAST_ID:
            self.PortHandler.is_using = False
            return rxpacket, result, error

        # set packet timeout
        if txpacket[PKT_INSTRUCTION] == INST_READ:
            self.PortHandler.setPacketTimeout(txpacket[PKT_PARAMETER0 + 1] + 6)
        else:
            self.PortHandler.setPacketTimeout(6)  # HEADER0 HEADER1 ID LENGTH ERROR CHECKSUM

        # rx packet
        while True:
            rxpacket, result = self.rxPacket()
            if result != COMM_SUCCESS or txpacket[PKT_ID] == rxpacket[PKT_ID]:
                break

        if result == COMM_SUCCESS and txpacket[PKT_ID] == rxpacket[PKT_ID]:
            error = rxpacket[PKT_ERROR]

        return rxpacket, result, error

    def ping(self, id):

        error = 0

        txpacket = [0] * 6

        if id > BROADCAST_ID:
            return COMM_NOT_AVAILABLE, error

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = 2
        txpacket[PKT_INSTRUCTION] = INST_PING

        rxpacket, result, error = self.txRxPacket(txpacket)

        return rxpacket, result, error

    def action(self, id):
        txpacket = [0] * 6

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = 2
        txpacket[PKT_INSTRUCTION] = INST_ACTION

        _, result, _ = self.txRxPacket(txpacket)

        return result

    def readData(self, id, address, length):
        txpacket = [0] * 8
        data = []

        if id > BROADCAST_ID:
            return data, COMM_NOT_AVAILABLE, 0

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = 4
        txpacket[PKT_INSTRUCTION] = INST_READ
        txpacket[PKT_PARAMETER0 + 0] = address
        txpacket[PKT_PARAMETER0 + 1] = length

        rxpacket, result, error = self.txRxPacket(txpacket)
        if result == COMM_SUCCESS:
            error = rxpacket[PKT_ERROR]

            data.extend(rxpacket[PKT_PARAMETER0: PKT_PARAMETER0 + length])

        return data, result, error

    def read1ByteData(self, id, address):
        data, result, error = self.readData(id, address, 1)
        data_read = data[0] if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def read2ByteData(self, id, address):
        data, result, error = self.readData(id, address, 2)
        data_read = self.makeWord16(data[0], data[1]) if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def read4ByteData(self, id, address):
        data, result, error = self.readData(id, address, 4)
        data_read = self.makeWord32(self.makeWord16(data[0], data[1]),
                                       self.makeWord16(data[2], data[3])) if (result == COMM_SUCCESS) else 0
        return data_read, result, error

    def writeDataOnly(self, id, address, length, data):
        txpacket = [0] * (length + 7)

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = length + 3
        txpacket[PKT_INSTRUCTION] = INST_WRITE
        txpacket[PKT_PARAMETER0] = address
        txpacket[PKT_PARAMETER0 + 1: PKT_PARAMETER0 + 1 + length] = data[0: length]

        result = self.txPacket(txpacket)
        self.PortHandler.is_using = False

        return result

    def writeReadData(self, id, address, length, data):
        txpacket = [0] * (length + 7)

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = length + 3
        txpacket[PKT_INSTRUCTION] = INST_WRITE
        txpacket[PKT_PARAMETER0] = address
        txpacket[PKT_PARAMETER0 + 1: PKT_PARAMETER0 + 1 + length] = data[0: length]
        rxpacket, result, error = self.txRxPacket(txpacket)

        return result, error

    def write1ByteDataOnly(self, id, address, data):
        data_write = [data]
        return self.writeDataOnly(id, address, 1, data_write)

    def writeRead1ByteData(self, id, address, data):
        data_write = [data]
        return self.writeReadData(id, address, 1, data_write)

    def write2ByteDataOnly(self, id, address, data):
        data_write = [self.getLowByte(data), self.getHighByte(data)]
        return self.writeDataOnly(id, address, 2, data_write)

    def writeRead2ByteData(self, id, address, data):
        data_write = [self.getLowByte(data), self.getHighByte(data)]
        return self.writeReadData(id, address, 2, data_write)

    def write4ByteDataOnly(self, id, address, data):
        data_write = [self.getLowByte(self.getLowWord32(data)),
                      self.getHighByte(self.getLowWord32(data)),
                      self.getLowByte(self.get_Highword32(data)),
                      self.getHighByte(self.get_Highword32(data))]
        return self.writeDataOnly(id, address, 4, data_write)

    def writeRead4ByteData(self, id, address, data):
        data_write = [self.getLowByte(self.getLowWord32(data)),
                      self.getHighByte(self.getLowWord32(data)),
                      self.getLowByte(self.get_Highword32(data)),
                      self.getHighByte(self.get_Highword32(data))]
        return self.writeReadData(id, address, 4, data_write)

    def regWriteTxOnly(self, id, address, length, data):
        txpacket = [0] * (length + 7)

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = length + 3
        txpacket[PKT_INSTRUCTION] = INST_REG_WRITE
        txpacket[PKT_PARAMETER0] = address

        txpacket[PKT_PARAMETER0 + 1: PKT_PARAMETER0 + 1 + length] = data[0: length]

        result = self.txPacket(txpacket)
        self.PortHandler.is_using = False

        return result

    def regWriteTxRx(self, id, address, length, data):
        txpacket = [0] * (length + 7)

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = length + 3
        txpacket[PKT_INSTRUCTION] = INST_REG_WRITE
        txpacket[PKT_PARAMETER0] = address

        txpacket[PKT_PARAMETER0 + 1: PKT_PARAMETER0 + 1 + length] = data[0: length]

        _, result, error = self.txRxPacket(txpacket)

        return result, error

    def syncReadTx(self, start_address, data_length, param, param_length):
        txpacket = [0] * (param_length + 8)
        # 8: HEADER0 HEADER1 ID LEN INST START_ADDR DATA_LEN CHKSUM

        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH] = param_length + 4  # 7: INST START_ADDR DATA_LEN CHKSUM
        txpacket[PKT_INSTRUCTION] = INST_SYNC_READ
        txpacket[PKT_PARAMETER0 + 0] = start_address
        txpacket[PKT_PARAMETER0 + 1] = data_length

        txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param[0: param_length]
        result = self.txPacket(txpacket)

        return result

    def syncReadRx(self, data_length, param_length):
        wait_length = (6 + data_length) * param_length
        self.PortHandler.setPacketTimeout(wait_length)
        rxpacket = []
        rx_length = 0
        while True:
            rxpacket.extend(self.PortHandler.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                result = COMM_SUCCESS
                break
            else:
                # check timeout
                if self.PortHandler.isPacketTimeout():
                    if rx_length == 0:
                        result = COMM_RX_TIMEOUT
                    else:
                        result = COMM_RX_CORRUPT
                    break
        self.PortHandler.is_using = False
        return result, rxpacket

    def syncWriteTxOnly(self, start_address, data_length, param, param_length):
        txpacket = [0] * (param_length + 8)
        # 8: HEADER0 HEADER1 ID LEN INST START_ADDR DATA_LEN ... CHKSUM

        txpacket[PKT_ID] = BROADCAST_ID
        txpacket[PKT_LENGTH] = param_length + 4  # 4: INST START_ADDR DATA_LEN ... CHKSUM
        txpacket[PKT_INSTRUCTION] = INST_SYNC_WRITE
        txpacket[PKT_PARAMETER0 + 0] = start_address
        txpacket[PKT_PARAMETER0 + 1] = data_length

        txpacket[PKT_PARAMETER0 + 2: PKT_PARAMETER0 + 2 + param_length] = param[0: param_length]

        _, result, _ = self.txRxPacket(txpacket)

        return result

    def reset(self, id):
        error = 0

        txpacket = [0] * 6

        if id > BROADCAST_ID:
            return COMM_NOT_AVAILABLE, error

        txpacket[PKT_ID] = id
        txpacket[PKT_LENGTH] = 2
        txpacket[PKT_INSTRUCTION] = INST_RESET

        rxpacket, result, error = self.txRxPacket(txpacket)

        return result, error

class GroupSyncRead:
    def __init__(self, handler, start_address, data_length):
        self.handler = handler
        self.start_address = start_address
        self.data_length = data_length

        self.last_result = False
        self.is_param_changed = False
        self.param = []
        self.data_dict = {}

        self.clearParam()

    def makeParam(self):
        if not self.data_dict:  # len(self.data_dict.keys()) == 0:
            return

        self.param = []

        for scs_id in self.data_dict:
            self.param.append(scs_id)

    def addParam(self, id):
        if id in self.data_dict:  # scs_id already exist
            return False

        self.data_dict[id] = []  # [0] * self.data_length

        self.is_param_changed = True
        return True

    def removeParam(self, id):
        if id not in self.data_dict:  # NOT exist
            return

        del self.data_dict[id]

        self.is_param_changed = True

    def clearParam(self):
        self.data_dict.clear()

    def txPacket(self):
        if len(self.data_dict.keys()) == 0:

            return COMM_NOT_AVAILABLE

        if self.is_param_changed is True or not self.param:
            self.makeParam()

        return self.handler.syncReadTx(self.start_address,
                                        self.data_length,
                                        self.param,
                                        len(self.data_dict.keys()))

    def rxPacket(self):
        self.last_result = True

        result = COMM_RX_FAIL

        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE

        result, rxpacket = self.handler.syncReadRx(self.data_length, len(self.data_dict.keys()))

        if len(rxpacket) >= (self.data_length + 6):
            for id in self.data_dict:
                self.data_dict[id], result = self.readRx(rxpacket, id, self.data_length)
                if result != COMM_SUCCESS:
                    self.last_result = False

        else:
            self.last_result = False

        return result

    def txRxPacket(self):
        result = self.txPacket()
        if result != COMM_SUCCESS:
            return result

        return self.rxPacket()

    def readRx(self, rxpacket, id, data_length):
        data = []
        rx_length = len(rxpacket)

        rx_index = 0
        while (rx_index + 6 + data_length) <= rx_length:
            headpacket = [0x00, 0x00, 0x00]
            while rx_index < rx_length:
                headpacket[2] = headpacket[1]
                headpacket[1] = headpacket[0]
                headpacket[0] = rxpacket[rx_index]
                rx_index += 1
                if (headpacket[2] == 0xFF) and (headpacket[1] == 0xFF) and headpacket[0] == id:

                    break

            if (rx_index+3+data_length) > rx_length:
                break
            if rxpacket[rx_index] != (data_length+2):
                rx_index += 1

                continue
            rx_index += 1
            Error = rxpacket[rx_index]
            rx_index += 1
            calSum = id + (data_length+2) + Error
            data = [Error]
            data.extend(rxpacket[rx_index : rx_index+data_length])
            for i in range(0, data_length):
                calSum += rxpacket[rx_index]
                rx_index += 1
            calSum = ~calSum & 0xFF

            if calSum != rxpacket[rx_index]:
                return None, COMM_RX_CORRUPT
            return data, COMM_SUCCESS

        return None, COMM_RX_CORRUPT

    def isAvailable(self, id, address, data_length):
        #if self.last_result is False or scs_id not in self.data_dict:
        if id not in self.data_dict:
            return False, 0

        if (address < self.start_address) or (self.start_address + self.data_length - data_length < address):
            return False, 0

        if not self.data_dict[id]:
            return False, 0

        if len(self.data_dict[id])<(data_length+1):
            return False, 0

        return True, self.data_dict[id][0]

    def getData(self, id, address, data_length):
        if data_length == 1:
            return self.data_dict[id][address-self.start_address+1]
        elif data_length == 2:
            return self.handler.makeWord16(self.data_dict[id][address-self.start_address+1],
                                self.data_dict[id][address-self.start_address+2])
        elif data_length == 4:
            return self.handler.makeWord32(self.handler.makeWord16(self.data_dict[id][address-self.start_address+1],
                                            self.data_dict[id][address-self.start_address+2]),
                                            self.handler.makeWord16(self.data_dict[id][address-self.start_address+3],
                                            self.data_dict[id][address-self.start_address+4]))
        else:
            return 0

class GroupSyncWrite:
    def __init__(self, handler, start_address, data_length):
        self.handler = handler
        self.start_address = start_address
        self.data_length = data_length

        self.is_param_changed = False
        self.param = []
        self.data_dict = {}

        self.clearParam()

    def makeParam(self):
        if not self.data_dict:
            return

        self.param = []

        for id in self.data_dict:
            if not self.data_dict[id]:
                return

            self.param.append(id)
            self.param.extend(self.data_dict[id])

    def addParam(self, id, data):
        if id in self.data_dict:  # id already exist
            return False

        if len(data) > self.data_length:  # input data is longer than set
            return False

        self.data_dict[id] = data

        self.is_param_changed = True
        return True

    def removeParam(self, id):
        if id not in self.data_dict:  # NOT exist
            return

        del self.data_dict[id]

        self.is_param_changed = True

    def changeParam(self, id, data):
        if id not in self.data_dict:  # NOT exist
            return False

        if len(data) > self.data_length:  # input data is longer than set
            return False

        self.data_dict[id] = data

        self.is_param_changed = True
        return True

    def clearParam(self):
        self.data_dict.clear()

    def txPacket(self):
        if len(self.data_dict.keys()) == 0:
            return COMM_NOT_AVAILABLE

        if self.is_param_changed is True or not self.param:
            self.makeParam()
        # print(self.data_dict)
        return self.handler.syncWriteTxOnly(self.start_address, self.data_length, self.param,
                                       len(self.data_dict.keys()) * (1 + self.data_length))