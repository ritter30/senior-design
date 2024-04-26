# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

# from kf import KF
from matplotlib.ticker import ScalarFormatter
from pyproj import Proj
from PIL import Image, ImageTk

# %%
# raw_data = np.loadtxt('/Users/pal/Desktop/senior_design/code/app/data/imu_and_gps_01.csv', delimiter=',', dtype=str)
raw_data = np.loadtxt('/Users/pal/Desktop/senior_design/code/app/data/imu_gps_gyro02.csv', delimiter=',', dtype=str)
raw_data = pd.DataFrame(raw_data)

# %%
imu_data = raw_data[raw_data[:][0].str.match(r'^-?\d*\.?\d*$')][:]
imu_data.columns = [
    'x_acceleration', 'y_acceleration', 'z_acceleration',
    'yaw', 'roll', 'pitch'
    ]
imu_data.reset_index(inplace=True)
imu_data['time'] = pd.to_datetime('2024-03-31 00:00:00.00', format='%Y-%m-%d %H:%M:%S.%f')
imu_data = imu_data.astype({
    'x_acceleration': np.float16,
    'y_acceleration': np.float16,
    'z_acceleration': np.float16
})

gps_data = raw_data.loc[raw_data[:][0].str.match(r'\d+:\d+.\d+'), :2]
gps_data.columns = ['time','latitude', 'longitude']
gps_data.drop_duplicates(subset='time', inplace=True)
gps_data.reset_index(inplace=True)
gps_data['idx_diff'] = 0
gps_data = gps_data.astype({
    'time': str,
    'latitude': str,
    'longitude': str,
    'idx_diff': int
})

# %%
gps_data.loc[0, 'time'] = pd.to_datetime(f'2024-03-31 00:{gps_data.loc[0, "time"]}', format='%Y-%m-%d %H:%M:%S.%f')

for i in range(gps_data.shape[0] - 1):
    gps_data.loc[i + 1, 'idx_diff'] = gps_data.loc[i + 1, 'index'] - gps_data.loc[i, 'index']
    gps_data.loc[i + 1, 'time'] = pd.to_datetime(f'2024-03-31 00:{gps_data.loc[i + 1, "time"]}', format='%Y-%m-%d %H:%M:%S.%f')
    gps_data.loc[i + 1, 'time_step'] = ((gps_data.loc[i + 1, 'time'] - gps_data.loc[i, 'time']) / gps_data.loc[i + 1, 'idx_diff']).total_seconds()


    # gps_data.loc[i + 1, 'time'] = gps_data.loc[i + 1, 'time']

# %%
orient_data = imu_data[['index','yaw','roll','pitch']]
orient_data = orient_data[500:]
orient_data.reset_index(inplace=True)

imu_data = imu_data.loc[500:, 'index':'z_acceleration']       # should really be renamed to accel_data
imu_data.reset_index(inplace=True)

# %%
chunks = []
start = 0
for i in range(1, gps_data.shape[0] - 1):
    chunk = slice(start, start + gps_data.loc[i + 1, 'idx_diff'])
    chunks.append(chunk)
    imu_data.loc[chunk, 'dt'] = gps_data['time_step'][i + 1]
    imu_data.loc[start, 'time'] = gps_data.loc[i, 'time']
    start += gps_data.loc[i + 1, 'idx_diff']
# %%

for chunk in chunks:
    i_ref = chunk.__getattribute__('start')
    t_ref = imu_data.loc[i_ref, 'time']
    for i in range(i_ref, chunk.__getattribute__('stop') - 1):
        imu_data.loc[i + 1, 'time'] = (imu_data.loc[i, 'time']) + pd.Timedelta(imu_data.loc[i, 'dt'], unit='s')
# %%
imu_data.dropna(inplace=True)
# %%
latest_meas = pd.Series(imu_data['time'].tail(1), index=range(gps_data.shape[0]))
gps_data = gps_data[gps_data['time'] < imu_data['time'].tail(1).squeeze()]    # get last time entry
# %%
out = gps_data[['latitude', 'longitude', 'time']]
# out.drop(index=0, axis=0, inplace=True)
out['latitude'] = out['latitude'].str.strip('N').astype(float)
out['longitude'] = out['longitude'].str.strip('W').astype(float)
gps_data = out
out.to_csv('/Users/pal/Desktop/senior_design/code/app/data/run01_gps.csv', header=False, index=False)
# %%

def meters_to_gps_coordinates(
        relative_distance_meters: np.array, 
        reference_latitude: float, 
        reference_longitude: float):
    # Earth's radius in meters
    earth_radius = 6371000

    # Convert relative distance to radians
    relative_distance_radians = relative_distance_meters / earth_radius

    # Calculate new latitude
    new_latitude = reference_latitude + (relative_distance_radians[1] * (180 / np.pi))

    # Calculate new longitude
    new_longitude = reference_longitude + (relative_distance_radians[0] * (180 / np.pi) / np.cos(reference_latitude * np.pi / 180))

    return new_latitude, new_longitude

def gps_dm_to_dd(lon, lat) -> np.array:
    dd_lat = ((lat / 100) % 1) / .6 + (lat // 100)
    dd_lon = (((lon / 100) % 1) / .6 + (lon // 100)) * -1

    return np.array([dd_lon, dd_lat])

# %%

mus = []
covs = []
x_accel = np.array([])
x_pos = np.array([])

H = np.zeros((3,9))

gps_times = gps_data['time']
imu_times = imu_data['time']

last_t = gps_times[1]

mus = []
covs = []
time = []

t = gps_data.loc[1, 'time']
dt = (t - last_t).total_seconds()
last_t = t
time.append(t)

# # H matrix for GPS
# H[0,0] = 1
# H[1,3] = 1
# H[2,6] = 1

# # R matrix for GPS
# R = np.array([
#     [10, 0, 0],
#     [0, 10, 0],
#     [0, 0, 10]
# ])

z_n = gps_data.loc[1, ['longitude', 'latitude']].to_numpy()
z_n = gps_dm_to_dd(z_n[0], z_n[1])
z_n = np.append(z_n, 0)          # just assuming 0 altitude for now

myProj = Proj(proj='utm', zone=16, ellps='WGS84', north=True, units='m')
utm_easting, utm_northing = myProj(z_n[0], z_n[1])
z_utm = np.array([[utm_easting, utm_northing, z_n[2]]]).T

# using GPS as initial state

init_state = np.array([[utm_easting,0,0,utm_northing,0,0,0,0,0]]).T

my_kf = KF(initial_state=init_state)

# print(my_kf)

covs.append(my_kf.cov)
mus.append(my_kf.mean)

i = 1
j = 2           # should reset the gps index to be more intuitive, but rn j=2 is where the start is

while i < imu_times.size and j < gps_times.size:
    # if i > 1000:
    #     break
    if imu_times[i] > gps_times[j - 1] and imu_times[i] < gps_times[j]:
        # print('Measuring IMU')
        # This is the logic for reading the IMU sensor data in between GPS pings
        # t = (imu_data.loc[i, 'time'].microsecond / 1000000)
        t = (imu_data.loc[i, 'time'])
        dt = (t - last_t).total_seconds()
        last_t = t
        time.append(t)


        # H matrix for IMU
        H[0,2] = 1
        H[1,5] = -1
        H[2,8] = 1

        # R matrix for IMU
        R = np.array([
            [50, 0, 0],
            [0, 50, 0],
            [0, 0, 50]
        ])

        z_n = imu_data.loc[i, ['y_acceleration', 'x_acceleration', 'z_acceleration']].to_numpy().reshape((3,1))

        i += 1

    else:
        # print('Measuring GPS')
        t = gps_data.loc[j, 'time']
        dt = (t - last_t).total_seconds()
        last_t = t
        time.append(t)

        # H matrix for GPS
        H[0,0] = 1
        H[1,3] = 1
        H[2,6] = 1

        # R matrix for GPS
        R = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ])

        z_n = gps_data.loc[j, ['longitude', 'latitude']].to_numpy()
        z_n = gps_dm_to_dd(z_n[0], z_n[1])
        z_n = np.append(z_n, 0)          # just assuming 0 altitude for now

        myProj = Proj(proj='utm', zone=16, ellps='WGS84', north=True, units='m')
        utm_easting, utm_northing = myProj(z_n[0], z_n[1])
        z_utm = np.array([[utm_easting, utm_northing, z_n[2]]]).T

        j += 1
        i += 1

    my_kf.predict(dt=dt)
    my_kf.update(
        meas_value=z_utm,
        meas_variance=R,
        meas_func=H
    )

    # print(my_kf)

    covs.append(my_kf.cov)
    mus.append(my_kf.mean)
#%%
raw_pos_data = gps_data.copy()
result = raw_pos_data.apply(
                    lambda row: 
                        myProj(*list(gps_dm_to_dd(row['longitude'], row['latitude']))),
                    axis=1
                    )
raw_pos_data['utm_eastings'] = [x[0] for x in result]
raw_pos_data['utm_northings'] = [x[1] for x in result]
# %%

utm_eastings = [float(mu[0][0]) for mu in mus]
utm_northings = [float(mu[3][0]) for mu in mus]

utm_coords = zip(utm_northings, utm_eastings)

gps_coords = []

for utm_coord in utm_coords:
    gps_coords.append(myProj(utm_coord[0], utm_coord[1], inverse=True)[::-1])

my_df = pd.DataFrame({
    'utm_eastings': utm_eastings,
    'utm_northings': utm_northings,
    'time': time,
})

# my_df = pd.DataFrame(gps_coords)
# my_df = pd.DataFrame(gps_coords[4000::50])
my_df.to_csv('/Users/pal/Desktop/senior_design/code/app/data/fused_run01.csv', index=False, header=False)

lower_lat = np.mean(utm_eastings) - 5
upper_lat = np.mean(utm_eastings) + 5

lat_lon_plot, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(9,6))

import matplotlib.dates as mdates

ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
ax[0].yaxis.set_major_formatter(ScalarFormatter())
ax[0].set_title('Position')
# ax[0].plot(time[:my_df.shape[0]], my_df.iloc[:,0], 'g')
ax[0].plot(time, utm_eastings, 'go', markersize=1)
ax[0].plot(raw_pos_data.time, raw_pos_data.utm_eastings, 'bo', markersize=1)
# ax[0].plot(my_df.iloc[:,1], my_df.iloc[:,0], 'o-')
# ax[0].ticklabel_format(useOffset=False, style='plain')
ax[0].fill_between(
    time,
    [float(mu[0][0] + 2*np.sqrt(cov[0,0])) for mu, cov in zip(mus,covs)],
    [float(mu[0][0] - 2*np.sqrt(cov[0,0])) for mu, cov in zip(mus,covs)],
    facecolor='r',
    alpha=0.5
    )

ax[0].set_xlim([time[3500], 19813.027083])
# ax[0].set_xlim([time[3500], time[7000]])
ax[0].set_ylim([5.074e5, 5.078e5])
ax[0].legend(['KF Longitude', 'GPS Longitude', 'Error'])

ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
ax[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=5.07e5))
# ax[1].set_title('Longitude')
ax[1].plot(time, utm_northings, 'go', markersize=1)
ax[1].plot(raw_pos_data.time, raw_pos_data.utm_northings, 'bo', markersize=1)
# ax[1].ticklabel_format(useOffset=False, style='plain')
ax[1].fill_between(
    time,
    [float(mu[3][0] + 2*np.sqrt(cov[3,3])) for mu, cov in zip(mus,covs)],
    [float(mu[3][0] - 2*np.sqrt(cov[3,3])) for mu, cov in zip(mus,covs)],
    facecolor='r',
    alpha=0.5
    )
ax[1].set_ylim([4.4751e6, 4.47553e6])
ax[1].legend(['KF Latitude', 'GPS Latitude', 'Error'])

# Calculate average error for latitude plot
avg_error_lat = np.mean([np.sqrt(cov[0,0]) for cov in covs])
# Calculate average error for longitude plot
avg_error_lon = np.mean([np.sqrt(cov[3,3]) for cov in covs])

# Add labels to subplots
ax[0].text(0.95, 0.05, f'Avg Error: {avg_error_lat:.2f}', transform=ax[0].transAxes, ha='right', va='bottom')
ax[1].text(0.95, 0.05, f'Avg Error: {avg_error_lon:.2f}', transform=ax[1].transAxes, ha='right', va='bottom')

# plt.show()

# %%
# live plot
data_offset = 4000

plotting_df = pd.DataFrame({
    'utm_easting': utm_eastings[data_offset:],
    'utm_northing': utm_northings[data_offset:],
    'time': time[data_offset:]
})

plotting_df = plotting_df.reset_index()

# Start the tkinter main loop
# root.mainloop()
# %%

utm_path, ax = plt.subplots(figsize=(8,7))
ax.plot(plotting_df.utm_easting, plotting_df.utm_northing, 'o-', markersize=1)
ax.set_xlabel('Longitude (UTM m)')
ax.set_ylabel('Latitude (UTM m)')

ax.set_title('Data Path in UTM Meters')

# %%
orient_data = orient_data.astype({
    'yaw': np.float16,
    'roll': np.float16,
    'pitch': np.float16
})
# %%
# Building Dashboard

def update():
    global index

    # Read a line from the serial port
    if index < (plotting_df.shape[0] - num_data_pts):

        # Read a row from the DataFrame
        rows = plotting_df.iloc[index: index + num_data_pts]

        ax[0].clear()
        ax[1].clear()

        ax[0].set_title('Live Plot of Position Using Kalman Filter')
        ax[0].set_xlabel('Latitude')
        ax[0].set_ylabel('Longitude')
        time_text = ax[0].text(0.65, 0.95, '', transform=ax[0].transAxes)
        time_text.set_text('Time: {}'.format(rows.time.iloc[-1]))

        x_pos = rows.utm_easting
        y_pos = rows.utm_northing

        # x_pos += 0.5 * row['dt'] ** 2 * row['x_accel']
        # y_pos += 0.5 * row['dt'] ** 2 * row['y_accel']

        # Append the data to the graph's data source
        # Assuming 'ax' is a matplotlib Axes object
        # ax.set_xlim([min(plotting_df.utm_easting), max(plotting_df.utm_easting)])
        # ax.set_ylim([min(plotting_df.utm_northing), max(plotting_df.utm_northing)])
        ax[0].plot(x_pos, y_pos, 'o-')

        avg_orientation = orient_data.iloc[index + num_data_pts - 10 : index + num_data_pts]['yaw'].mean()

        avg_orientation = np.radians(avg_orientation)

        avg_orientation_corrected = 3 * np.pi / 2 - avg_orientation

        # print(avg_orientation_corrected)

        u = -np.cos(avg_orientation_corrected)
        v = np.sin(avg_orientation_corrected)

        ax[1].quiver(0,0,u,v, angles='xy', scale_units='xy', scale=1)

        ax[1].set_xlim(-1, 1)
        ax[1].set_ylim(-1, 1)

        # Update the graph
        canvas.draw()

        # Increment the current index
        index += 1

    # Call the update function again after 1000 milliseconds
    root.after(10, update)

index = 0
num_data_pts = 200

root = tk.Tk()

root.geometry('1600x900+50+50')
root.configure(bg='#2e2e2e')

root.iconbitmap('/Users/pal/Desktop/senior_design/code/app/assets/favicon.icns')

notebook = ttk.Notebook(root)

frame01 = tk.Frame(notebook)
frame02 = tk.Frame(notebook)
frame_gps = tk.Frame(notebook, width=800, height=600)
frame_gps.pack_propagate(False)

# Add the frames to the Notebook
notebook.add(frame01, text='Accuracy Analysis')
notebook.add(frame02, text='Live Plot')
notebook.add(frame_gps, text='GPS Compare')

fig, ax = plt.subplots(1, 2, figsize=(16,9))
gs = GridSpec(1, 2, width_ratios=[3,1])

ax[0] = fig.add_subplot(gs[0])
ax[1] = fig.add_subplot(gs[1])

# Scroll Bar
canvas_frame02 = tk.Canvas(frame02)
scrollbar_frame02 = ttk.Scrollbar(frame02, orient="vertical", command=canvas_frame02.yview)

# Create a frame to add to the canvas
scrollable_frame02 = ttk.Frame(canvas_frame02)

# Tell the canvas to scroll the frame
canvas_frame02.configure(yscrollcommand=scrollbar_frame02.set, scrollregion=canvas_frame02.bbox("all"))

# Add the scrollable frame to the canvas
canvas_frame02.create_window((0, 0), window=scrollable_frame02, anchor="nw")

# Update the scrollregion of the canvas when the size of the frame changes
def on_frame02_configure(event):
    canvas_frame02.configure(scrollregion=canvas_frame02.bbox("all"))

scrollable_frame02.bind("<Configure>", on_frame02_configure)

canvas_frame02.pack(side="left", fill="both", expand=True)
scrollbar_frame02.pack(side='right', fill='y')

# Your content goes here

## Frame 01 setup

canvas_fr01 = FigureCanvasTkAgg(lat_lon_plot, master=frame01)
canvas_fr01.draw()

canvas_fr01.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas_fr01, frame01)
toolbar.update()
canvas_fr01.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

## Frame 02 setup

# Create a canvas for the plot
canvas = FigureCanvasTkAgg(fig, master=scrollable_frame02)
canvas.draw()

canvas.get_tk_widget().pack()

start_button = tk.Button(scrollable_frame02, text='Start', command=update)
start_button.pack()

## GPS Frame
canvas_gps = FigureCanvasTkAgg(utm_path, master=frame_gps)
canvas.draw()

canvas_gps.get_tk_widget().grid(row=0, column=1)

toolbar_gps = NavigationToolbar2Tk(canvas_gps, frame_gps)
toolbar_gps.update()
canvas_gps.get_tk_widget().grid(row=0, column=1, sticky='s')

image = Image.open('/Users/pal/Desktop/senior_design/code/app/graphics/my_route_run01.png')
image = image.resize((800, 700))
photo = ImageTk.PhotoImage(image)

image_label = tk.Label(frame_gps, image=photo)
image_label.image = photo

image_label.grid(row=0, column=0)

# Pack the Notebook
notebook.pack(expand=True, fill='both')

# Start the tkinter main loop
root.mainloop()
# %%
