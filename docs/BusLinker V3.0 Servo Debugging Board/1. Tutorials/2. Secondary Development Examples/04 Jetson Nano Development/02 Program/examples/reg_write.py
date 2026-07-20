import sys
import os

sys.path.append("..")

from servo_sdk import *

# Initialize PortHandler instance
# ex)
# Windows: "COM1"
# Linux: "/dev/ttyUSB0"
# Mac: "/dev/tty.usbserial-*"
PortHandler = PortHandler('COM13') #
ServoHandler = HxServoHandler(PortHandler)

# Open port
if PortHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()


# Set port baudrate 1000000
if PortHandler.setBaudRate(1000000):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

while True:
    # Servo (ID1~10) runs at a maximum speed ofV=1500*0.059=88.5rpm until it reaches position P1=1000
    result, error = ServoHandler.writeRegPosEx(1, 1000, 1500, 0)
    if result != COMM_SUCCESS:
        print("%s" % ServoHandler.getTxRxResult(result))
        break

    if error != 0:
        print("%s" % ServoHandler.getRxPacketError(error))
        break
    ServoHandler.regAction()
    time.sleep(((1000-20) / 1500) + 0.1)  #//[(P1-P0)/(V)] + 0.1

    # Servo (ID1~10) runs at a maximum speed ofV=1500*0.059=88.5rpm until it reaches position P0=20
    result, error = ServoHandler.writeRegPosEx(1, 20, 1500, 0)
    if result != COMM_SUCCESS:
        print("%s" % ServoHandler.getTxRxResult(result))
        break

    if error != 0:
        print("%s" % ServoHandler.getRxPacketError(error))
        break

    ServoHandler.regAction()
    time.sleep(((1000 - 20) / 1500) + 0.1)
    # time.sleep((1000-20)/(1500) + 0.1)) #//[(P1-P0)/(V)] + 0.1

# Close port
PortHandler.closePort()