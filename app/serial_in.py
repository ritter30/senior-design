import serial

def readUART(port, baud):
    # port = '/dev/tty.usbserial-0001'
    # baud = 9600

    ser = serial.Serial(port, baud)

    try:
        line = ser.readline().decode().strip()

        return line

    except KeyboardInterrupt:
        ser.close()

if __name__ == '__main__':
    while True:
        print(readUART('/dev/tty.usbserial-0001', 9600))