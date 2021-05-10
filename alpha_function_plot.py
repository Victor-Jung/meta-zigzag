import numpy as np
import matplotlib.pyplot as plt

temperature = 0.05
rho = 0.999
step = 0

x = np.arange(-0.4, 0, 0.0001)
y = np.exp(x / temperature)

plt.figure(1)
plt.title("Alpha function depending of the Temperature")
plt.plot(x, y, label="T = " + str(format(temperature, ".3f")) + " step = " + str(step))

for i in range(4):
    temperature = temperature * (rho**500)
    y = np.exp(x / temperature)
    step += 500
    plt.plot(x, y, label="T = " + str(format(temperature, ".3f")) + " step = " + str(step))

plt.legend(loc="upper left")
plt.xlabel("U_y - U_x")
plt.ylabel("alpha")
plt.show()