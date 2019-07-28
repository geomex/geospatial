'''
Utilities for Kristin's Project

by: Joel A. Gongora
date: 05/30/2019
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.interpolate import griddata
from matplotlib import colors

# Define Imputer #
imp = Imputer(strategy = 'mean')

def gen_plots(df, cols):
    '''
    Description:

    This function will generate kernel density functions and
    scatter plots for fieldData.csv

    '''
    
    if 'x' in cols:
        x_coord = 'xxcoords'
        y_coord = 'xycoords'
    else:
        x_coord = 'yxcoords'
        y_coord = 'yycoords'
        
    # Drop NaNs if needed --------------------
    if any(df[cols].isna()):
        
        df_tmp = df.dropna(
            how='all',
            subset=[cols]
        )
        
    else:
        df_tmp = df.copy()
        
    # Plot Kernel Density of xReal --------------------
    plt.figure()
    plt.subplot(211)
    sns.distplot(
        df_tmp[cols].values,
        kde_kws = {
            "color": 'k',
            "lw": 3,
            "label": "KDE"
        }
    )
    plt.title(
        'PDF of {}'.format(cols)
    )
    plt.tight_layout()
    
    # Plot Scatter of xReal --------------------
    
    cmap = sns.cubehelix_palette(
        dark=0.3,
        light=0.8,
        as_cmap=True
    )
    
    plt.subplot(212)
    sns.scatterplot(
        x=x_coord,
        y=y_coord,
        hue=cols,
        size=cols,
        data=df_tmp,
        palette=cmap,
    )
    plt.title(
        '{} Scatter'.format(cols)
    )
    plt.tight_layout()
    plt.savefig(
        './figures/scatter_density_{}.png'.format(cols)
    )


# --------------------------------- #
# Plotting Imagesc by Gridding Data #
# --------------------------------- #

def gen_mesh_plot(df, cols):
    '''
    Description:

    This function will generate mesh plots for the fieldData.csv

    '''
    if 'x' in cols:
        x_coord = 'xxcoords'
        y_coord = 'xycoords'
    else:
        x_coord = 'yxcoords'
        y_coord = 'yycoords'
        
    # Drop NaNs if needed --------------------
    
    df_tmp = df[
        (0==df.isna().sum(axis=1))
    ]

    # Min and Max ------------------------------

    x_lim = [df_tmp[x_coord].min(), df_tmp[x_coord].max()]
    y_lim = [df_tmp[y_coord].min(), df_tmp[y_coord].max()] 

    xi = np.linspace(x_lim[0], x_lim[1], 100)
    yi = np.linspace(y_lim[0], y_lim[1], 100)
    XX, YY = np.meshgrid(
        xi,
        yi
    )
    
    grid_z = griddata(
        df_tmp[[x_coord, y_coord]].values,
        df_tmp[cols].values,
        (XX, YY),
        method='nearest'
    )
    
    dx = (xi[1] - xi[0])/2
    dy = (yi[1] - yi[0])/2
    extent = [
        xi[0] - dx,
        xi[-1] - dx,
        yi[0] - dy,
        yi[-1] - dy
    ]
    
    plt.figure()
    plt.imshow(
        imp.fit_transform(np.flipud(grid_z)),
        extent=extent,
        cmap='jet',
        interpolation='bicubic',
    )
    plt.title(
        '{} Gridded Plot'.format(cols)
    )
    plt.colorbar()
    plt.savefig(
        './figures/gridded_image_{}.png'.format(cols)
    )
