#%%
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt

#%%
df = pd.read_csv('./data/sine_wave.csv')

print(df.head())

# %%
sin_df = pd.read_csv('./data/sine_wave.csv')

sin_iter = iter(sin_df.itertuples(index=True))

data = pd.DataFrame(columns=['t', 'sin'])

time = [None] * 100
sin = [None] * 100

# while True:
for i in range(100):
        try:
            index, t, value = next(sin_iter)

            # print(t, value)
            # time[i] = dt.datetime.now().timestamp()
            time[i] = t + 2*np.pi
            sin[i] = value

            if index == sin_df.shape[0] - 1:
                sin_iter = iter(sin_df.itertuples(index=True))

        except StopIteration:
            sin_iter = iter(sin_df.itertuples(index=True))

data = pd.DataFrame({
     't': time,
     'sin': sin
})

sin_df.iloc[:-100,:] = sin_df.iloc[100:, :] # slide data to the left
sin_df.iloc[-100:,:] = data # postfix new data to front

print(sin_df)

# %%
sin_df.to_csv("./data/sine_wave.csv", index=False)
# %%
