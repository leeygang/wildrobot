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

result, error = ServoHandler.writePosEx(1, -3000, 100, 0)

if result != COMM_SUCCESS:
    print("%s" % ServoHandler.getTxRxResult(result))
if error != 0:
    print("%s" % ServoHandler.getRxPacketError(error))


# Close port
PortHandler.closePort()