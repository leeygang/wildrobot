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
    # Add Servo id （1~2） goal position\moving speed\moving accc value to the Syncwrite parameter storage
    # Servo (ID1~2) runs at a maximum speed of V=1500*0.059=88.5rpm until it reaches position P1=1000
    for id in range(1, 3):
        add_param_result = ServoHandler.syncWritePosEx(id, 1000, 1500, 0)
        if add_param_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % id)

    # Syncwrite goal position
    result = ServoHandler.GroupSyncWrite.txPacket()
    if result != COMM_SUCCESS:
        print("%s" % result.getTxRxResult(result))
        break

    time.sleep(((1000-20) / 1500) + 0.1)  #//[(P1-P0)/(V)] + 0.1

    # Clear syncwrite parameter storage
    ServoHandler.GroupSyncWrite.clearParam()

    for id in range(1, 3):
        add_param_result = ServoHandler.syncWritePosEx(id, 20, 1500, 0)
        if add_param_result != True:
            print("[ID:%03d] groupSyncWrite addparam failed" % id)

    # Syncwrite goal position
    result = ServoHandler.GroupSyncWrite.txPacket()
    if result != COMM_SUCCESS:
        print("%s" % result.getTxRxResult(result))
        break

    time.sleep(((1000 - 20) / 1500) + 0.1)
    # time.sleep((1000-20)/(1500) + 0.1)) #//[(P1-P0)/(V)] + 0.1

    # Clear syncwrite parameter storage
    ServoHandler.GroupSyncWrite.clearParam()
# Close port
PortHandler.closePort()