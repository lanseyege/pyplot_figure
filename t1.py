#sample figure

import numpy as np
import matplotlib.pyplot as plt

def get_pos():
    a = [1.0, 1.0, 2.0, 2.0, 3.0]
    b = [3.0, 4.0, 4.0, 5.0, 7.0]
    return a, b
xp, yp = get_pos()
plt.plot(xp, yp, 'bs', yp, xp, 'g^')
plt.axis([0, 10, 0, 10])
plt.show()
