import yaml
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

'''
with open("test/go_prob_AlexNet_S1000_I100_R100-2001.yaml") as f:
    data_doc = yaml.safe_load(f)

x = data_doc['number_of_iter']
y = data_doc['proba_list']

plt.plot(x, y, 'D--', label='mcmc')
plt.title("Reliability of MCMC depending on the number of iteration")
plt.xlabel("Number of iteration")
plt.ylabel("Probability of reaching GO")
plt.legend(loc='upper left')
plt.savefig("test/go_prob_AlexNet_S1000_I100_R100-2001.png")
'''