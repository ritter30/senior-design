import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import serial
import csv

# Create a Serial object to read from the serial port
ser = serial.Serial('/dev/tty.usbserial-A50285BI', 9600)

# Create a Tk object for the main window
root = tk.Tk()

# Create a Figure object for the graph
fig, ax = plt.subplots()

# Create a FigureCanvasTkAgg object to embed the graph in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Create an update function
def update():
    # Read a line from the serial port
    line = ser.readline()

    # Parse the line as CSV
    reader = csv.reader([line])
    for row in reader:
        data = list(map(float, row))

    # Append the data to the graph's data source
    ax.plot(data)

    # Update the graph
    canvas.draw()

    # Call the update function again after 1000 milliseconds
    root.after(1000, update)

# Call the update function after 1000 milliseconds
root.after(1000, update)

# Start the tkinter main loop
root.mainloop()