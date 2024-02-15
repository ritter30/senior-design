import serial
# import time

def readUART(port, baud):
    # port = '/dev/tty.usbserial-0001'
    # baud = 9600

    ser = serial.Serial(port, baud)

    try:
        line = ser.readline().decode()

        return line

    except KeyboardInterrupt:
        ser.close()

if __name__ == '__main__':
    while True:
        print(readUART('/dev/tty.usbserial-A50285BI', 9600))
        # time.sleep(500)