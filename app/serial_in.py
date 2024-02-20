import serial
# import time

def readUART(port, baud):
    # port = '/dev/tty.usbserial-0001'
    # baud = 9600

    ser = serial.Serial(port, baud)

    try:
        line = ser.readline().decode().strip()

        return line

    except KeyboardInterrupt:
        ser.close()

def parse_data(data):
    # Split data into individual values
    values = data.decode().split(",")

    # Convert and return orientation values (modify based on your data format)
    roll = float(values[0])
    pitch = float(values[1])
    yaw = float(values[2])
    return roll, pitch, yaw

def update(frame):
    # Read data from serial port
    data = ser.readline().strip()

    # Parse data
    roll, pitch, yaw = parse_data(data)

    # Update plot data (modify based on your visualization preference)
    x_data.append(frame)
    y_data.append(roll)
    z_data.append(pitch)

    # Clear and redraw plot
    plt.cla()
    plt.plot(x_data, y_data, label="Roll")
    plt.plot(x_data, z_data, label="Pitch")
    plt.legend()
    plt.title("Live Orientation Data")
    plt.xlabel("Time Frame")
    plt.ylabel("Angle (Degrees)")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    # while True:
    #     print(readUART('/dev/tty.usbserial-A50285BI', 9600))
    #     # time.sleep(500)

    # Set serial port and baud rate
    port = "/dev/tty.usbserial-A50285BI"  # Replace with your port
    baudrate = 9600  # Replace with your baud rate

    # Open serial port
    ser = serial.Serial(port, baudrate)

    # Initial data and plot setup
    x_data = []
    y_data = []
    z_data = []

    fig, ax = plt.subplots()

    # Animate
    ani = animation.FuncAnimation(fig, update, interval=100)

    # Display plot
    plt.show()