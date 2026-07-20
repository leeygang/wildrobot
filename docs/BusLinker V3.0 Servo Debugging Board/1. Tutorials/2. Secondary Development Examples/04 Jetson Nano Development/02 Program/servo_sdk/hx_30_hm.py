from .protocol_paket_handler import *

#baudrate define
HX_30HM_1M = 0
HX_30HM_0_5M = 1
HX_30HM_250K = 2
HX_30HM_128K = 3
HX_30HM_115200 = 4
HX_30HM_76800 = 5
HX_30HM_57600 = 6
HX_30HM_38400 = 7

#Memory table

#NVS
HX_30HM_SERVO_MAIN_VERSION = 3
HX_30HM_SERVO_SEC_VERSION = 4
HX_30HM_ID = 5
HX_30HM_BAUD_RATE = 6
HX_30HM_CW_DEAD = 26
HX_30HM_CCW_DEAD = 27
HX_30HM_POS_OFFSET_L = 31
HX_30HM_POS_OFFSET_H = 32
HX_30HM_MODE = 33

#SRAM

#Write read
HX_30HM_TORQUE_ENABLE = 40
HX_30HM_ACC = 41
HX_30HM_GOAL_POSITION_L = 42
HX_30HM_GOAL_POSITION_H = 43
HX_30HM_PWM_SPEED_L = 44
HX_30HM_PWM_SPEED_H = 45
HX_30HM_GOAL_SPEED_L = 46
HX_30HM_GOAL_SPEED_H = 47
HX_30HM_MAX_TORQUE_L = 48
HX_30HM_MAX_TORQUE_H = 49

#Only read
HX_30HM_PRESENT_POSITION_L = 56
HX_30HM_PRESENT_POSITION_H = 57
HX_30HM_PRESENT_SPEED_L = 58
HX_30HM_PRESENT_SPEED_H = 59
HX_30HM_PRESENT_LOAD_L = 60
HX_30HM_PRESENT_LOAD_H = 61
HX_30HM_PRESENT_VOLTAGE = 62
HX_30HM_PRESENT_TEMPERATURE = 63
HX_30HM_MOVING_STATUS = 66
HX_30HM_PRESENT_CURRENT_L = 69
HX_30HM_PRESENT_CURRENT_H = 70


class HxServoHandler(PacketHandler):
    def __init__(self, PortHandler):
        PacketHandler.__init__(self, PortHandler, 0)
        self.GroupSyncWrite = GroupSyncWrite(self, HX_30HM_ACC, 7)

    def selectMode(self, id, mode):
        _mode = [max(0, min(mode, 3))]
        return self.writeReadData(id, HX_30HM_MODE, 1, _mode)

    def torqueEnable(self, id):
        state = [1]
        return self.writeReadData(id, HX_30HM_TORQUE_ENABLE, 1, state)

    def torqueDisable(self, id):
        state = [0]
        return self.writeReadData(id, HX_30HM_TORQUE_ENABLE, 1, state)

    def writeCurPosOffset(self, id):
        state = [128]
        return self.writeReadData(id, HX_30HM_TORQUE_ENABLE, 1, state)

    def writePosOffset(self, id, offset):
        _offset = max(-2047, min(offset, 2047))
        _offset = self.toServo(_offset, 11)
        txpacket = [self.getLowByte(_offset), self.getHighByte(_offset)]

        return self.writeReadData(id, HX_30HM_POS_OFFSET_L, len(txpacket), txpacket)

    def writeAcc(self, id, acc):
        _acc = [max(0, min(acc, 254))]
        return self.writeReadData(id, HX_30HM_ACC, 1, _acc)

    def writeSpeed(self, id, speed):
        _speed = max(0, min(speed, 3400))
        txpacket = [self.getLowByte(_speed), self.getHighByte(_speed)]
        return self.writeReadData(id, HX_30HM_GOAL_SPEED_L, len(txpacket), txpacket)

    def writePos(self, id, pos):
        _pos = max(-30719, min(pos, 30719))
        _pos = self.toServo(_pos, 15)
        txpacket = [self.getLowByte(_pos), self.getHighByte(_pos)]
        return self.writeReadData(id, HX_30HM_GOAL_POSITION_L, len(txpacket), txpacket)

    def writePosEx(self, id, pos, speed, acc):
        _pos = max(-30719, min(pos, 30719))
        _speed = max(0, min(speed, 3400))
        _acc = max(0, min(acc, 254))

        _pos = self.toServo(_pos, 15)

        txpacket = [_acc,
                    self.getLowByte(_pos), self.getHighByte(_pos),
                    0,0,
                    self.getLowByte(_speed), self.getHighByte(_speed)]

        return self.writeReadData(id, HX_30HM_ACC, len(txpacket), txpacket)

    def writePwmSpeed(self, id, speed):
        _speed = max(-1000, min(speed, 1000))
        _speed = self.toServo(_speed, 10)
        txpacket = [self.getLowByte(_speed), self.getHighByte(_speed)]
        return self.writeReadData(id, HX_30HM_PWM_SPEED_L, len(txpacket), txpacket)

    def writeMaxTorque(self, id, max_torque):
        _max_torque = max(0, min(max_torque, 1000))
        print(_max_torque)
        txpacket = [self.getLowByte(_max_torque), self.getHighByte(_max_torque)]
        return self.writeReadData(id, HX_30HM_MAX_TORQUE_L, len(txpacket), txpacket)

    def writeRegPosEx(self, id, pos, speed, acc):
        _pos = max(-30719, min(pos, 30719))
        _speed = max(0, min(speed, 3400))
        _acc = max(0, min(acc, 254))
        _pos = self.toHost(_pos, 15)
        txpacket = [_acc,
                    self.getLowByte(_pos), self.getHighByte(_pos),
                    0,0,
                    self.getLowByte(_speed), self.getHighByte(_speed)]
        return self.regWriteTxRx(id, HX_30HM_ACC, len(txpacket), txpacket)

    def regAction(self):
        return self.action(BROADCAST_ID)

    def syncWritePosEx(self, id, pos, speed, acc):
        _pos = max(-30719, min(pos, 30719))
        _speed = max(0, min(speed, 3400))
        _acc = max(0, min(acc, 254))
        txpacket = [_acc,
                    self.getLowByte(_pos), self.getHighByte(_pos),
                    0,0,
                    self.getLowByte(_speed), self.getHighByte(_speed)]
        return self.GroupSyncWrite.addParam(id, txpacket)

    def readPosOffset(self, id):
        offset, result, error = self.read2ByteData(id, HX_30HM_POS_OFFSET_L)
        # The representation of negative numbers
        return -(~(offset - 1) & 0xFFFF) if offset > 2047 else offset, result, error

    def readPos(self, id):
        present_position, result, error = self.read2ByteData(id, HX_30HM_PRESENT_POSITION_L)
        return self.toHost(present_position, 15), result, error

    def readSpeed(self, id):
        present_speed, result, error = self.read2ByteData(id, HX_30HM_PRESENT_SPEED_L)
        return self.toHost(present_speed, 15), result, error

    def readPosSpeed(self, id):
        present_position_speed, result, error = self.read4ByteData(id, HX_30HM_PRESENT_POSITION_L)
        present_position = self.getLowWord32(present_position_speed)
        present_speed = self.get_Highword32(present_position_speed)
        return self.toHost(present_position, 15), self.toHost(present_speed, 15), result, error

    def readTemperature(self, id):
        present_temp, result, error = self.read1ByteData(id, HX_30HM_PRESENT_TEMPERATURE)
        return present_temp, result, error

    def readVoltage(self, id):
        present_vol, result, error = self.read1ByteData(id, HX_30HM_PRESENT_VOLTAGE)
        return present_vol, result, error

    def readCurrent(self, id):
        present_cur, result, error = self.read2ByteData(id, HX_30HM_PRESENT_CURRENT_L)
        return present_cur, result, error

    def readLoad(self, id):
        load, result, error = self.read2ByteData(id, HX_30HM_PRESENT_LOAD_L)
        return self.toHost(load, 10), result, error

    def readMoving(self, id):
        moving, result, error = self.read1ByteData(id, HX_30HM_MOVING_STATUS)
        return moving, result, error



