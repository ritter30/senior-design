# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from kf import KF
from matplotlib.ticker import ScalarFormatter
from pyproj import Proj
from IPython.display import HTML
from kf import KF

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
# from kf import KF

init_state = np.array([[0,0,0,0,0,0,0,0,0]]).T

mus = []
covs = []
x_accel = np.array([])
x_pos = np.array([])

H = np.zeros((3,9))

my_kf = KF(initial_state=init_state)

gps_times = gps_data['time']
imu_times = imu_data['time']

last_t = gps_times[1]

mus = []
covs = []
time = []

# need to start with a gps coordinate ***
print('Measuring GPS')

t = gps_data.loc[1, 'time']
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

z_n = gps_data.loc[1, ['longitude', 'latitude']].to_numpy()
z_n = gps_dm_to_dd(z_n[0], z_n[1])
z_n = np.append(z_n, 0)          # just assuming 0 altitude for now

myProj = Proj(proj='utm', zone=16, ellps='WGS84', north=True, units='m')
utm_x, utm_y = myProj(z_n[0], z_n[1])
z_utm = np.array([[utm_x, utm_y, z_n[2]]]).T

my_kf.predict(dt=dt)
my_kf.update(
    meas_value=z_utm,
    meas_variance=R,
    meas_func=H
)

# print(my_kf)

covs.append(my_kf.cov)
mus.append(my_kf.mean)

i = 1
j = 2           # should reset the gps index to be more intuitive, but rn j=2 is where the start is

while i < imu_times.size and j < gps_times.size:
    # if i > 1000:
    #     break
    if imu_times[i] > gps_times[j - 1] and imu_times[i] < gps_times[j]:
        print('Measuring IMU')
        # This is the logic for reading the IMU sensor data in between GPS pings
        # t = (imu_data.loc[i, 'time'].microsecond / 1000000)
        t = (imu_data.loc[i, 'time'])
        dt = (t - last_t).total_seconds()
        last_t = t
        time.append(t)


        # H matrix for IMU
        H[0,2] = 1
        H[1,5] = 1
        H[2,8] = 1

        # R matrix for IMU
        R = np.array([
            [1000, 0, 0],
            [0, 1000, 0],
            [0, 0, 1000]
        ])

        z_n = imu_data.loc[i, 'x_acceleration':'z_acceleration'].to_numpy().reshape((3,1))

        i += 1

    else:
        print('Measuring GPS')
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
        utm_x, utm_y = myProj(z_n[0], z_n[1])
        z_utm = np.array([[utm_x, utm_y, z_n[2]]]).T

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

# %%

utm_ys = [float(mu[0][0]) for mu in mus]
utm_xs = [float(mu[3][0]) for mu in mus]

utm_coords = zip(utm_ys, utm_xs)

gps_coords = []

for utm_coord in utm_coords:
    gps_coords.append(myProj(utm_coord[0], utm_coord[1], inverse=True)[::-1])

my_df = pd.DataFrame(gps_coords)
my_df.to_csv('/Users/pal/Desktop/senior_design/code/app/data/fused_run01.csv', index=False, header=False)

lower_lat = np.mean(utm_xs) - 5
upper_lat = np.mean(utm_xs) + 5

fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(16,9))

import matplotlib.dates as mdates

ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
ax[0].yaxis.set_major_formatter(ScalarFormatter())
ax[0].set_title('Posistion')
# ax[0].plot(time[:my_df.shape[0]], my_df.iloc[:,0], 'g')
ax[0].plot(time, utm_xs, 'g')
# ax[0].plot(my_df.iloc[:,1], my_df.iloc[:,0], 'o-')
# ax[0].ticklabel_format(useOffset=False, style='plain')
ax[0].fill_between(
    time,
    [float(mu[3][0] + 2*np.sqrt(cov[0,0])) for mu, cov in zip(mus,covs)],
    [float(mu[3][0] - 2*np.sqrt(cov[0,0])) for mu, cov in zip(mus,covs)],
    facecolor='r',
    alpha=0.5
    )

ax[0].set_xlim([19813.024306, 19813.027083])
ax[0].set_ylim([4.4751e6, 4.47553e6])
ax[0].legend(['Latitude', 'Error'])

ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
ax[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=5.07e5))
# ax[1].set_title('Longitude')
ax[1].plot(time, utm_ys, 'b')
# ax[1].ticklabel_format(useOffset=False, style='plain')
ax[1].fill_between(
    time,
    [float(mu[0][0] + 2*np.sqrt(cov[3,3])) for mu, cov in zip(mus,covs)],
    [float(mu[0][0] - 2*np.sqrt(cov[3,3])) for mu, cov in zip(mus,covs)],
    facecolor='r',
    alpha=0.5
    )
ax[1].set_ylim([5.074e5, 5.078e5])
ax[1].legend(['Longitude', 'Error'])

# Calculate average error for latitude plot
avg_error_lat = np.mean([np.sqrt(cov[0,0]) for cov in covs])
# Calculate average error for longitude plot
avg_error_lon = np.mean([np.sqrt(cov[3,3]) for cov in covs])

# Add labels to subplots
ax[0].text(0.95, 0.05, f'Avg Error: {avg_error_lat:.2f}', transform=ax[0].transAxes, ha='right', va='bottom')
ax[1].text(0.95, 0.05, f'Avg Error: {avg_error_lon:.2f}', transform=ax[1].transAxes, ha='right', va='bottom')

plt.show()
# %%
# Create a new figure and an axis
fig, ax = plt.subplots()

# Initialize a line object that will be updated
line, = ax.plot([], [], 'o')

# Function to initialize the plot
def init():
    mean_x = np.mean(utm_xs)
    mean_y = np.mean(utm_ys)
    std_x = np.std(utm_xs)
    std_y = np.std(utm_ys)

    ax.set_xlim(mean_x - 1, mean_x + 1)  # Set x-axis limits
    ax.set_ylim(mean_y - 1, mean_y + 1)  # Set y-axis limits
    return line,

# Function to update the plot
def update(frame):
    # Here you should add your logic to get the new coordinates
    # For this example, we're just using the frame number as x and y
    x = utm_xs[frame * 100]
    y = utm_ys[frame * 100]

    line.set_data(x, y)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(len(utm_xs) // 100), init_func=init, blit=True)

ani.save('/Users/pal/Desktop/senior_design/code/app/animations/fused_run01_animation.mp4', writer='ffmpeg')

HTML(ani.to_jshtml())

# %%
