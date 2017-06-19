#draw svm example figure

import numpy as np
import matplotlib.pyplot as plt

#the margin points
k1 = 1.5
k2 = 3.5
k3 = 2.5
#move the point ~
def lin(a, b):
    return abs(a-b+1)
def lln(a, b):
    return abs(a-b-1)
#positive points
def get_p():
    a = [1.0, 3.0]
    b = [2.0, 4.0]
    A = np.random.sample(20)*3+1
    B = np.random.sample(20)*3+2
    A = np.insert(A, [0, 0], [1.0, 3.0])
    B = np.insert(B, [0, 0], [2.0, 4.0])
    for i in range(len(A)):
        if A[i] + 1.4 > B[i]:
            k = lin(A[i], B[i])
            B[i] += k
            A[i] -= k
    return A, B
#negative points
def get_n():
    a = [3.0]
    b = [2.0]
    A = np.random.sample(20)*4+1
    B = np.random.sample(20)*4
    A = np.insert(A, [0], [3.0])
    B = np.insert(B, [0], [2.0])
    for i in range(len(A)):
        if A[i] -1.5 < B[i]:
            k = lln(A[i], B[i])
            A[i] += k
            B[i] -= k
            
    return A, B
#draw circle
Cx = np.array([1.0,3.0,3.0])
Cy = np.array([2.0,4.0,2.0])
plt.plot(Cx,Cy, marker=r'$\bigodot$', markersize=12,
        linewidth=0, alpha=0.3, color='k', label='Gost-Cell')
#get random points
xp, yp = get_p()
xn, yn = get_n()
plt.plot(xp, yp, 'bs', xn, yn, 'r^')
plt.plot(np.arange(0,6),'black')
#(x1,y1)=zip(*[(1.0,2.0),(k1,k1)])
#(x2,y2)=zip(*[(3.0,4.0),(k2,k2)])
#(x3,y3)=zip(*[(3.0,2.0),(k3,k3)])
#plt.plot(x1,y1, x2, y2, x3, y3)
plt.axis([-1.0, 5.5, -1.0, 5.5])
plt.text(1.1,1.6,'r')
plt.text(3.1,3.6,'r')
plt.text(2.6,2.1,'r')
#draw margin line
plt.annotate('', xy=(1.0,2.0),xytext=(1.52,1.48),arrowprops=dict(edgecolor='blue',arrowstyle="<->"),)
plt.annotate('', xy=(3.0,4.0),xytext=(3.52,3.48),arrowprops=dict(edgecolor='blue',arrowstyle="<->"),)
plt.annotate('', xy=(3.0,2.0),xytext=(2.48,2.52),arrowprops=dict(edgecolor='r',arrowstyle="<->"),)
#draw axes with arraw
plt.annotate('', xy=(0.0,-1.0),xytext=(0.0,5.5),arrowprops=dict(edgecolor='black',arrowstyle="<-"),)
plt.annotate('', xy=(-1.0,0.0),xytext=(5.5,0.0),arrowprops=dict(edgecolor='black',arrowstyle="<-"),)
#draw circle
#s1 = np.arange(0.5,1.5,0.1)
#s2 = np.arange(2.5,3.5,0.1)
#s1, s2 = np.meshgrid(s1,s2)
#plt.contour(s1, s2, (s1-1.0)**2 + (s2-3.0)**2, [4])
#draw dash
w1 = np.arange(0.0,5,0.1)
w2 = np.array([e*0.8+0.5 for e in w1])
plt.plot(w1, w2, 'r--')
w1 = np.arange(0.0,5,0.1)
w2 = np.array([e*1.2-0.5 for e in w1])
plt.plot(w1, w2, 'r--')
#hidden original axes 
axes = plt.subplot(111)
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.spines['bottom'].set_color('none')
axes.spines['left'].set_color('none')
#axes.spines['bottom'].set_position(('data',0))
#axes.spines['left'].set_position(('data',0))
axes.set_xticks([])
axes.set_yticks([])
#show figure
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
