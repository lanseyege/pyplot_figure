#draw the figure about clustering and classification

import numpy as np
import matplotlib.pyplot as plt

def get_A1():
    A = np.random.sample(10)*1.5+0.4
    B = np.random.sample(10)*2+0.5
    return A,B
def get_B1():
    A = np.random.sample(10)*1.5+0.4
    B = np.random.sample(10)*2+2.8
    return A,B
def get_A2():
    A = np.random.sample(10)*1.5+3.5
    B = np.random.sample(10)*2+0.8
    return A,B
def get_B2():
    A = np.random.sample(10)*1.5+3.5
    B = np.random.sample(10)*2+2.9
    return A,B

xp, yp = get_A1()
xo, yo = get_B1()
xn, yn = get_A2()
xm, ym = get_B2()

plt.plot(xp, yp, 'b*')
plt.plot(xo, yo, 'r*')
plt.plot(xn, yn, 'b*')
plt.plot(xm, ym, 'r*')

plt.axis([-1.0, 5.5, -1.0, 5.5])
for i in range(len(xp)):
    plt.text(xp[i], yp[i], 'B',color='black')
    plt.text(xo[i], yo[i], 'A',color='red')
    plt.text(xn[i], yn[i], 'B',color='black')
    plt.text(xm[i], ym[i], 'A',color='red')
s1 = np.arange(0.0, 5.5, 0.1)
s2 = np.array([e*0.1+2.5 for e in s1])
plt.plot(s1, s2, 'b')
#circle
n1 = np.arange(-1.0, 2.5, 0.1)
n2 = np.arange(-1.0, 5.5, 0.1)
n1, n2 = np.meshgrid(n1, n2)
dash = plt.contour(n1, n2, (n1-1.25)**2/1.3+(n2-2.625)**2/7-1, 0, linestyles='dashed')
#for d in dash.collections:
#    d.set_dashes([(0,(2.0,2.0))])
n1 = np.arange(2.5, 5.5, 0.1)
n2 = np.arange(-1.0, 5.5, 0.1)
n1, n2 = np.meshgrid(n1, n2)
dash = plt.contour(n1, n2, (n1-4.1)**2/1.3+(n2-2.625)**2/7-1, 0, linestyles='dashed')
#for d in dash.collections:
#    d.set_dashes([(0,(2.0,2.0))])
#axes arrow
plt.annotate('', xy=(0.0,-1.0),xytext=(0.0,5.5),arrowprops=dict(edgecolor='black',arrowstyle="<-"),)
plt.annotate('', xy=(-1.0,0.0),xytext=(5.5,0.0),arrowprops=dict(edgecolor='black',arrowstyle="<-"),)
#hidden axes
axes = plt.subplot(111)
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.spines['bottom'].set_color('none')
axes.spines['left'].set_color('none')
#axes.spines['bottom'].set_position(('data',0))
#axes.spines['left'].set_position(('data',0))
axes.set_xticks([])
axes.set_yticks([])

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

