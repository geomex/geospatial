'''This script will generate a meshplot to visualize the Real
and Imaginary components of Signal. 

by: Joel A. Gongora
date: May 30, 2019
'''
# ------------------------ #
# Import Necessary Modules #
# ------------------------ #

import pandas as pd
import matplotlib.pyplot as plt
from utils import (gen_plots, gen_mesh_plot)

df = pd.read_csv('./fieldData.csv')
df.columns.tolist()

# --------------------------------------- #
# Plot Kernel Densities and Scatter Plots #
# --------------------------------------- #

gen_plots(df,'xReal')
gen_plots(df,'yReal')
gen_plots(df,'xImag')
gen_plots(df,'yImag')

# ------------------ #
# Plot Gridded Data  #
# ------------------ #

gen_mesh_plot(df,'xReal')
gen_mesh_plot(df,'yReal')
gen_mesh_plot(df,'xImag')
gen_mesh_plot(df,'yImag')
plt.show()
