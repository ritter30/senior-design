import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

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

imu_data = imu_data.iloc[(imu_data.shape[0] // 2):]

x_pos = []
y_pos = []
time = []
prev_x = 0
prev_y = 0

fig, ax = plt.subplots()

def animate(i, time, x_pos, y_pos, prev_x, prev_y):
    row = imu_data.iloc[i]

    time.append(row['time'])

    x_pos.append(prev_x + 0.5 * row['dt'] ** 2 * row['x_accel'])
    y_pos.append(prev_y + 0.5 * row['dt'] ** 2 * row['y_accel'])

    # print(len(x_pos))

    prev_x = x_pos[-1]
    prev_y = y_pos[-1]

    time = time[-50:]
    x_pos = x_pos[-50:]
    y_pos = y_pos[-50:]

    y_min = min(y_pos)
    y_max = max(y_pos)

    ax.clear()
    ax.plot(time, y_pos)
    # ax.plot(x_pos, y_pos)
    # ax.set_xlim([0,1e-4])
    # ax.set_xlim([time[0],])
    ax.set_ylim([y_min,y_max])

ani = FuncAnimation(
    fig, animate, 
    fargs=(time, x_pos, y_pos, prev_x, prev_y), 
    interval=10, repeat=True
    )

plt.show()