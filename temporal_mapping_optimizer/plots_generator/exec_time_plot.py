from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
import numpy as np

# AlexNet L3 with hint driven spatial unrolling C/K 
# Su capacity : Row 16 : Col 16

# 300 represents number of points to make between T.min and T.max
xnew = np.linspace(6, 11, 300)

lpf_range = np.arange(6, 11, 1)
sa_loma_time = [1.656454610824585, 1.656454610824585, 1.656454610824585, 1.656454610824585, 1.656454610824585]
loma_time = [0.21014904975891113, 0.7168512344360352, 5.186602354049683, 52.02914237976074, 535.9377088546753]
meta_loma_time = [0.21014904975891113, 0.7168512344360352, 1.656454610824585, 1.656454610824585, 1.656454610824585]

spl = make_interp_spline(lpf_range, loma_time, k=1)  # type: BSpline
power_smooth = spl(xnew)

plt.title("AlexNet Layer 3 with Hint Driven C/K")
plt.yscale('log')
plt.xlabel("Lpf Limit")
plt.ylabel("Time (s)")
plt.plot(lpf_range, sa_loma_time, label='Simulated Annealing', color='tab:orange')
plt.plot(lpf_range, loma_time, label='Loma', color='tab:blue')
plt.plot(lpf_range, meta_loma_time, '--', linewidth=2, label='Loma', color='tab:green')
plt.legend(loc='upper left')
plt.show()