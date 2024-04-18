# %%
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import serial
import csv
import pandas as pd

# Create a Serial object to read from the serial port
# ser = serial.Serial('/dev/tty.usbserial-A50285BI', 9600)

# Create a Tk object for the main window
root = tk.Tk()

# Create a Figure object for the graph
fig, ax = plt.subplots()

# Create a FigureCanvasTkAgg object to embed the graph in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

gps_data = pd.read_csv('/Users/pal/Desktop/senior_design/code/app/data/run01_gps.csv', 
                       header=None, 
                       index_col=0
                       ).reset_index()
imu_data = pd.read_csv('/Users/pal/Desktop/senior_design/code/app/data/run01_imu.csv', 
                       header=None, 
                       index_col=0
                       ).reset_index()

imu_data.drop(columns=[0, 1], inplace=True)

gps_data.columns = ['lat', 'lon', 'time']
imu_data.columns = ['x_accel', 'y_accel', 'z_accel', 'dt', 'time']

index = 0
x_pos = 0
y_pos = 0

# %%
# Create an update function
def update():
    global index
    global x_pos
    global y_pos

    # Read a line from the serial port
    if index < len(imu_data):
        # Read a row from the DataFrame
        row = imu_data.iloc[index]

        x_pos += 0.5 * row['dt'] ** 2 * row['x_accel']
        y_pos += 0.5 * row['dt'] ** 2 * row['y_accel']

        # Append the data to the graph's data source
        # Assuming 'ax' is a matplotlib Axes object
        ax.plot(x_pos, y_pos)

        # Update the graph
        canvas.draw()

        # Increment the current index
        index += 1

    # Call the update function again after 1000 milliseconds
    root.after(1, update)

root.after(10, update)

# Start the tkinter main loop
root.mainloop()