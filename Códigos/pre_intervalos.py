# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:45:54 2024

@author: corir
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.ticker as plticker
import seaborn as sbs

sbs.set_context("paper")

times = np.arange(0, 101)
# data1 = [-1.16014651, -0.71369354, -1.29547375, -0.20699523, -0.67443581, -1.05167094, -0.56718779, -1.22553315, -0.07898293, -0.59082237, -0.97924173, -0.53935062, -0.94570218, -1.90159707, -0.75895689, -1.11568475,  0.895432  , -0.48553561,  0.18397471, -1.39443811, -1.0004462 , -0.78102187, -1.01617117,  0.57869139, -1.3757193 , -0.87427282, -0.77883443, -0.99983981, -0.14946146, -0.14725738, -0.7558269 , -0.55665446, -1.91121369, -0.72074521, -0.65981093, -0.70015561, -1.10498571, -1.0408903 ,  0.47300337, -0.69412317, -1.5797519 , -0.50932713, -0.28719511, -1.13881346, -0.7406333 , -1.65304511, -1.40351318, -0.17259265, -1.43542826, -1.63277685, -0.03276488, -0.30813917, -1.52692586, -0.99558497, -0.58063622, 0.89009557, -1.53125   ,  0.7677719 ]
# data2 = [-0.99457571,  1.40502851, -0.25634787, -0.68009521,  0.95025527, -0.98058068, -0.35953291, -0.13533889, -0.85867141, -0.75070469, -1.04757418, -1.12429455, -0.04252793,  0.31013746,  1.64631468, -1.28957128,  0.06209089, -0.10935028, -1.31927015, -1.14197202, -0.97991187, -1.45228495,  1.22051514,  1.41067686, -1.11755803, 0.05626552, -0.75540465, -0.62177703,  0.1039224 , -0.81794348, 0.94703408, -1.49528249, -0.16153198,  1.55713699, -1.05649091, -0.32388284,  0.26713704, -0.8387457 , -0.36994029, -0.66498419, -0.08787496, -0.80532258, -1.36403819, -0.72469826, -1.28170603, -1.44610572,  1.11703826, -0.52245752, -1.69737063, -1.23393724, 1.58539955, -1.36303476,  0.89393019]
# data3 = data1
# data4 = data2
# data5 = data1
# data6 = data2
# data7 = data1
# data8 = data2

datas = [data1, data2, data3, data4]

def interval(datas):
    min_datas = []
    max_datas = []
    means = []
    std_data = []
    
    for i in range(0, len(datas)):
        min_datas.append(min(datas[i]))
        max_datas.append(max(datas[i]))
        means.append(np.mean(datas[i]))
        std_data.append(np.std(datas[i]))
            
    return min_datas, max_datas, means, std_data



def bool2extreme(mask, times) :
    """return xmins and xmaxs for intervals in times"""
    binary = 1*mask
    slope = np.diff(binary)

    extr = (slope != 0)
    signs = slope[extr]
    mins = list(times[1:][slope==1])
    maxs = list(times[:-1][slope==-1])
    if signs[0]==-1:
        mins = [times[0]] + mins
    if signs[-1]==1:
        maxs = maxs + [times[-1]]
    return mins, maxs

def plot_interval(mask, times, xlim=None, y=0, thickness=0.4, color='k', ax=None):
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_position(('data', 0))
    ax.tick_params(labelbottom=True)  # to get tick labels on all axes
    # ax.tick_params(which='both', direction='in')`  # tick marks above instead below the axis
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1)) # major ticks in steps of 10
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))  # minor ticks in steps of 1
    ax.set_ylim(-1.5,.5)
    if xlim is None:
        xlim = (times[0]-0.9, times[-1]+0.9)
    ax.set_xlim(xlim)
    xmins, xmaxs = bool2extreme(mask, times)
    for xmin, xmax in zip(xmins, xmaxs):
        #ax.add_patch(Rectangle((xmin, y-thickness), xmax-xmin, 2*thickness, linewidth=0, color=color))
        ax.add_patch(Rectangle((xmin, y), xmax-xmin, thickness, linewidth=0, color=color))
    
    triangle1 = [(xlim[0]*1.05, y), (xlim[0], y-thickness), (xlim[0], y+thickness)]
    ax.add_patch(Polygon(triangle1, linewidth=0, color='black', clip_on=False))
    triangle2 = [(xlim[1]*1.05, y), (xlim[1], y-thickness), (xlim[1], y+thickness)]
    ax.add_patch(Polygon(triangle2, linewidth=0, color='black', clip_on=False))
    return ax

def plot_interval(min0, min1, max0, max1, mean, std, xlim=None, y=0, thickness=0.4, color1='k', color2 = 'k', alpha = 0.7, ax=None, time2 = "up"):
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_position(('data', 0))
    ax.tick_params(labelbottom=True)  # to get tick labels on all axes
    # ax.tick_params(which='both', direction='in')`  # tick marks above instead below the axis
    #ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1)) # major ticks in steps of 10
    #ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))  # minor ticks in steps of 1
    ax.set_ylim(-1.5,.5)
    if xlim is None:
        max_tot = max(max0, max1)
        min_tot = min(min0, min1)
        interval_data = max_tot - min_tot
        xlim =  (min_tot-interval_data*0.1, max_tot+interval_data*0.1)
    ax.set_xlim(xlim)
    ax.add_patch(Rectangle((min0, y), max0-min0, thickness, linewidth=0, color=color1, alpha = alpha))
    ax.plot(mean[0], thickness/2, "o", color = darken_color(color1))
    if time2 == "up":
        ax.add_patch(Rectangle((min1, y), max1-min1, thickness, linewidth=0, color= color2, alpha = alpha))
        ax.plot(mean[1], thickness/2, "o", color = darken_color(color2))
    else:
        #se grafica por abajo del eje x
        ax.add_patch(Rectangle((min1, y-thickness), max1-min1, thickness, linewidth=0, color= color2, alpha = alpha))
        ax.plot(mean[1], -thickness/2, "o", color = darken_color(color2))
    triangle1 = [(xlim[0] - (xlim[1]-xlim[0])*0.03, y), (xlim[0], y-thickness), (xlim[0], y+thickness)]
    ax.add_patch(Polygon(triangle1, linewidth=0, color='black', clip_on=False))
    triangle2 = [(xlim[1] + (xlim[1]-xlim[0])*0.03, y), (xlim[1], y-thickness), (xlim[1], y+thickness)]
    ax.add_patch(Polygon(triangle2, linewidth=0, color='black', clip_on=False))
    return ax
#%%
# n_plots = len(masks)
# dist_between_axis_in_inches = 0.4

# fig, axs = plt.subplots(n_plots, sharex=True, figsize=(10, dist_between_axis_in_inches*len(masks)))
# for i, mask in enumerate(masks) :
#     axs[i] = plot_interval(mask, times, xlim=(times[0]-0.5, times[-1]+0.5), ax=axs[i], color='lime')
# axs[-1].set_xlabel('Timeaaaa (min)', ha = 'right')
# axs[-1].xaxis.set_label_coords(-0.02, 0.97)
# axs[-2].set_xlabel('coridorid', ha = 'right')
# axs[-2].xaxis.set_label_coords(-0.02, 0.97)
# plt.show()
datas =  [data1, data2, data3, data4]#, data5, data6, data7, data8]
min_datas, max_datas, means, std_data = interval(datas)

n_plots = int(len(datas)/2)
dist_between_axis_in_inches = 0.4

fig, axs = plt.subplots(n_plots, sharex=False, figsize=(10, dist_between_axis_in_inches*len(datas)))
axs[0].set_xlabel(var1)
axs[1].set_xlabel(var2)
#axs[2].set_xlabel(var3)
#axs[3].set_xlabel(var4)
for i in range(0, int(len(datas)/2)):
    axs[i] = plot_interval(min_datas[i*2], min_datas[i*2+1], max_datas[i*2], max_datas[i*2+1], means[i*2:i*2+2], std_data[i*2:i*2+2], thickness=0.4, ax=axs[i], color1= color_1, color2 = darkened_color_1, time2 = "down")
    axs[i].xaxis.set_label_coords(1.1, 0.97)

plt.tight_layout()
plt.show()

#axs[-2].set_xlabel('coridorid', ha = 'right')
#axs[-2].xaxis.set_label_coords(-0.03, 0.97)
plt.show()
#%%

mask = [masks[0], masks[1]]
times = x_data
x_lim =  xlim=(x_data[0]*1.1, x_data[-1]*1.1)
color = 'lime'
y=0
thickness=0.4
ax=None

plt.figure(8), plt.clf()
if ax is None:
    ax = plt.gca()
ax.yaxis.set_visible(False)
ax.spines['left'].set_color('None')
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.spines['bottom'].set_position(('data', 0))
ax.tick_params(labelbottom=True)  # to get tick labels on all axes
# ax.tick_params(which='both', direction='in')`  # tick marks above instead below the axis
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1)) # major ticks in steps of 10
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))  # minor ticks in steps of 1
ax.set_ylim(-1.5,.5)
if xlim is None:
    xlim = (times[0]-0.9, times[-1]+0.9)
ax.set_xlim(xlim)
xmins, xmaxs = bool2extreme(mask[0], times)
xmins1, xmaxs1 = bool2extreme(mask[1], times)
#for xmin, xmax in zip(xmins, xmaxs):
    #se grafica por abajo del eje x
    #ax.add_patch(Rectangle((xmin, y-thickness), xmax-xmin, thickness, linewidth=0, color=color))
    #se grafica por arriba del eje x
    #ax.add_patch(Rectangle((xmin, y), xmax-xmin, thickness, linewidth=0, color=color))
ax.add_patch(Rectangle((xmins[0], y), xmaxs[0]-xmins[0], thickness, linewidth=0, color=color_1, alpha = 0.7))
ax.add_patch(Rectangle((xmins1[0], y), xmaxs1[0]-xmins1[0], thickness, linewidth=0, color= darkened_color_1, alpha = 0.7))

triangle1 = [(xlim[0]*1.05, y), (xlim[0], y-thickness), (xlim[0], y+thickness)]
ax.add_patch(Polygon(triangle1, linewidth=0, color='black', clip_on=False))
triangle2 = [(xlim[1]*1.05, y), (xlim[1], y-thickness), (xlim[1], y+thickness)]
ax.add_patch(Polygon(triangle2, linewidth=0, color='black', clip_on=False))


plt.show()