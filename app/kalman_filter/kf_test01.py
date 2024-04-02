# %%
import numpy as np
import pandas as pd

# %%
raw_data = np.loadtxt('/Users/pal/Desktop/senior_design/code/app/data/imu_and_gps_01.csv', delimiter=',', dtype=str)
raw_data = pd.DataFrame(raw_data)

# %%
imu_data = raw_data[raw_data[:][0].str.match(r'^-?\d*\.?\d*$')][:]
imu_data.columns = ['x_acceleration', 'y_acceleration', 'z_acceleration']
imu_data.reset_index(inplace=True)
imu_data['time'] = pd.to_datetime('2024-03-31 00:00:00.00', format='%Y-%m-%d %H:%M:%S.%f')
imu_data = imu_data.astype({
    'x_acceleration': np.float16,
    'y_acceleration': np.float16,
    'z_acceleration': np.float16
})

gps_data = raw_data[raw_data[:][0].str.match(r'\d+:\d+.\d+')][:]
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
imu_data = imu_data[116:]
imu_data.reset_index(inplace=True)

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
gps_data = gps_data[gps_data['time'] < pd.to_datetime('2024-03-31 00:20:01.00', format='%Y-%m-%d %H:%M:%S.%f')]
# %%
out = gps_data[['latitude', 'longitude', 'time']]
out.drop(index=0, axis=0, inplace=True)
out['latitude'] = out['latitude'].str.strip('N').astype(float)
out['longitude'] = out['longitude'].str.strip('W').astype(float)
out.to_csv('/Users/pal/Desktop/senior_design/code/app/data/run01_gps.csv', header=False, index=False)
# %%
