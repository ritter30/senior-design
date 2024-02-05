import numpy as np
import pandas as pd

def gen_sine_wave():
    t = np.linspace(0, 2*np.pi, 1000)
    sin = np.sin(t)

    df = pd.DataFrame({
        't': t,
        'sin': sin
    })

    df.to_csv('./data/sine_wave.csv')

if __name__ == '__main__':
    gen_sine_wave()