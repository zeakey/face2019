import numpy as np
import matplotlib.pyplot as plt

it = np.arange(25000)


def lr(it, gamma):
  return 1 - np.exp(- it * gamma)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].plot(lr(it, 0.001))
axes[0, 0].set_title("$\\gamma$="+str(0.001))

axes[0, 1].plot(lr(it, 0.0025))
axes[0, 1].set_title(0.0025)

axes[0, 2].plot(lr(it, 0.005))
axes[0, 2].set_title(0.005)


axes[1, 0].plot(lr(it, 0.01))
axes[1, 0].set_title(0.01)

axes[1, 1].plot(lr(it, 0.025))
axes[1, 1].set_title(0.025)

axes[1, 2].plot(lr(it, 0.05))
axes[1, 2].set_title(0.05)

plt.suptitle('lr=$e^{(-iter \\times \\gamma)}$', fontsize=16)
plt.savefig('gamma-vs-lr.pdf')

