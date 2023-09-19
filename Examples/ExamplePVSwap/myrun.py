from ipywidgets import IntSlider, FloatText
import pandas as pd
import numpy as np
import glob
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.font_manager import FontProperties


from scipy.stats.kde import gaussian_kde
from numpy.linalg.linalg import LinAlgError


ore_exe = "C:\\Users\\fhoefer\\source\\repos\\ore\\build\\App\\Release\\ore.exe"
xml = "C:\\Users\\fhoefer\\source\\repos\\ore\\Examples\\ExamplePVSwap\\Input\\ore.xml"
#subprocess.call([ore_exe, xml])

npv_cube_filename = "C:\\Users\\fhoefer\\source\\repos\\ore\\Examples\\Example_fh\\Output\\rawcube.csv"
df = pd.read_csv(npv_cube_filename)
df.columns = ['Id', 'NettingSet', 'DateIndex',
              'Date', 'Sample', 'Depth', 'Value']

print("Netting sets:")
allNettingSets = df.NettingSet.unique()
print(allNettingSets)

print("TradeIds:")
AllTradeIds = df.Id.unique()
print(AllTradeIds)

nettingSet = allNettingSets[0]
tradeId = AllTradeIds[3]

df_surface = df[(df.NettingSet == nettingSet) & df.Id.isin([tradeId])][[
    'Id', 'Value', 'Sample', 'DateIndex']].groupby(['DateIndex', 'Sample']).sum().reset_index()

print(df.head())

print(df_surface[:100])

fig_surface = plt.figure()
fig_surface.canvas.set_window_title('Density Surface')

dates = pd.to_datetime(df['Date']).unique()
dates = dates - dates.min()
years = dates.astype('timedelta64[D]') / np.timedelta64(1, 'D') / 365

global df_data
grid_size = 50
npv_min = df_surface.Value.min()
npv_max = df_surface.Value.max()
dist_space = np.linspace(npv_min, npv_max, grid_size)
num_dates = len(df_surface.DateIndex.unique())
density_values = np.zeros((num_dates, grid_size))
for k in range(num_dates):
    row = df_surface.loc[df_surface['DateIndex'] == k, 'Value'].values
    # try:
    histogramm = np.histogram(row, bins=dist_space)
    density_values[k][:-1] = histogramm[0]
    #density = gaussian_kde(row)
    #density_values[k] = density(dist_space)
    # except:
    #    density_values[k] = np.zeros(grid_size)
density_max = np.max(density_values)


fig_surface.clear()
ax_surface = fig_surface.add_subplot(111, projection='3d')
date_step = 4
ax_surface.set_xticks(years[::date_step])
ax_surface.set_xticklabels(years[::date_step])
ax_surface.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: "{:,.0f}".format(x)))
ax_surface.set_xlabel('years')
ax_surface.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: "{:,.1f}".format(x/1000000)))
ax_surface.set_ylabel("exposure_selector.value"+" [mn]")
X, Y = np.meshgrid(years, dist_space)
ax_surface.plot_trisurf(X.flatten(), Y.flatten(),
                        density_values.T.flatten(), cmap=cm.jet, linewidth=0)
# plt.show()
plt.savefig(
    "C:\\Users\\fhoefer\\source\\repos\\ore\\Examples\\Example_fh\\Output\\" + tradeId+'.png')


fig_surface = plt.figure()
ax_time_slider = fig_surface.add_subplot(111)
fig_surface.canvas.set_window_title('Time Slider')


def plot_time_slider(change):
    date = change['new']
    ax_time_slider.cla()
    ax_time_slider.plot(
        dist_space, density_values[date], color='k', label=exposure_selector.value)
    ax_time_slider.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax_time_slider.set_xlim(npv_min, npv_max)
    ax_time_slider.set_ylim(0, density_max)
    ax_time_slider.set_xlabel("Exposure")
    year_text.value = "{:,.3f}".format(years[date])


time_slider = IntSlider(min=0, max=num_dates-1,
                        value=0, description='DateIndex:')
time_slider.observe(plot_time_slider, names='value')
year_text = FloatText(description="Years:", value=years[time_slider.value])
display(HBox([time_slider, year_text]))
