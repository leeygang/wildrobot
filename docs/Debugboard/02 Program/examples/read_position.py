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
    pos, result, error = ServoHandler.readLoad(1)
    if result != COMM_SUCCESS:
        print("%s" % ServoHandler.getTxRxResult(result))
        break

    if error != 0:
        print("%s" % ServoHandler.getRxPacketError(error))
        break

    print(pos)
    time.sleep(0.1)  # Delay for 0.1 seconds(延迟 0.1 秒)

PortHandler.closePort()