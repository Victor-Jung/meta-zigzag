from scipy.interpolate import make_interp_spline, BSpline
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.pyplot as plt
import numpy as np

# AlexNet L3 with hint driven spatial unrolling C/K 
# Su capacity : Row 16 : Col 16

# 300 represents number of points to make between T.min and T.max
xnew = np.linspace(6, 11, 300)

lpf_range = np.arange(6, 11, 1)
sa_loma_time = [1.656454610824585, 1.656454610824585, 1.656454610824585, 1.656454610824585, 1.656454610824585]
loma_time = [0.21014904975891113, 0.7168512344360352, 5.186602354049683, 52.02914237976074, 535.9377088546753]
timeloop_time = [273, 273, 273, 273, 273]
meta_loma_time = [0.21014904975891113, 0.7168512344360352, 1.656454610824585, 1.656454610824585, 1.656454610824585]
meta_loma_time2 = [0.21014904975891113, 0.7168512344360352, 1.656454610824585, 1.63663, 1.656454610824585, 1.656454610824585]

spl = make_interp_spline(lpf_range, loma_time, k=1)  # type: BSpline
power_smooth = spl(xnew)

fig, ax = plt.subplots()

plt.yscale('log')
plt.xlabel("Number of LPFs", fontsize=15, labelpad=10)
plt.ylabel("Time (s)", fontsize=15, labelpad=10)
plt.xticks([6,7,8,9,10], [6,7,8,9,10], fontsize=13)
plt.yticks(fontsize=13)

plt.plot(lpf_range, sa_loma_time, 'v-', linewidth=3, label='Simulated Annealing', color='tab:orange')
plt.plot(lpf_range, loma_time, 'o-', linewidth=3, label='Exhaustive', color='tab:blue')
plt.plot(lpf_range, meta_loma_time, '*', markersize=12, label='SALSA', color='tab:green')
plt.plot(lpf_range, timeloop_time, 'P-', linewidth=3, label='Timeloop', color='tab:cyan')
plt.plot([6,7,7.415,8,9,10], meta_loma_time2, linewidth=3, color='tab:green')

plt.legend(loc='upper left', framealpha=1, ncol=4, edgecolor='grey', fontsize=13)

ax.set_facecolor('#F2F2F2')
ax.tick_params(axis='y', which='major', pad=5)
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='--')
ax.set_axisbelow(True)

plt.show()