import sys
import os

sys.path.append("..")

from servo_sdk import *


# Initialize PortHandler instance
# ex)
# Windows: "COM1"
# Linux: "/dev/ttyUSB0"
# Mac: "/dev/tty.usbserial-*"
PortHandler = PortHandler('/dev/ttyACM0') #
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

# try to ping the ID:1 Servo
# Get receive data
rxpacket, result, error = ServoHandler.ping(1)
if result != COMM_SUCCESS:
    print("%s" % ServoHandler.getTxRxResult(result))
else:
    print(f"rx: {' '.join(f'{byte:02X}' for byte in rxpacket)}")
if error != 0:
    print("%s" % ServoHandler.getRxPacketError(error))



