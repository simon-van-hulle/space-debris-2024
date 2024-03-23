import os
import numpy as np
import matplotlib.pyplot as plt

# np.set_printoptions(linewidth=150, suppress=True, threshold=np.inf, formatter={"float": "{: 0.3E}".format})
# np.set_printoptions(linewidth=150, suppress=True, threshold=np.inf, precision=20)
np.set_printoptions(formatter={"float": "{: 20.12f}".format}, linewidth=250, threshold=np.inf)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
stylefile = os.path.join(CURRENT_PATH, 'myBmh.mplstyle')
plt.style.use(stylefile)


COLORS =  plt.rcParams['axes.prop_cycle'].by_key()['color']
COLOR =  lambda i: COLORS[i % len(COLORS)]

MARKERS = ['.', 'o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X', '8', '1', '2', '3', '4', '+', 'x', '|', '_']
MARKER = lambda i: MARKERS[i % len(MARKERS)]
