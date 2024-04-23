# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# from kf import KF
from matplotlib.ticker import ScalarFormatter
from pyproj import Proj

post_kf_data = pd.read_csv('/Users/pal/Desktop/senior_design/code/app/data/fused_run01.csv', header=None)


# %%
