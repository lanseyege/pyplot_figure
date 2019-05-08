import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize

path = './save/CV_10_ones.grad'
grad = pickle.load(open(path, 'rb'))

path = './save/CV_10_ones.data'
data = pickle.load(open(path, 'rb'))
print(data)
(n, m) = data.shape

def sigm(x):
    return 1.0/(1.0 + np.exp(-x)) - 0.5

t = 2
xrans = np.arange(m)
#data2, grad2 = data, grad
data2, grad2 = np.transpose(data), np.transpose(grad)
signs = np.sign(grad2)
temp = np.abs(grad2)
#grad2 = -0.0001/np.log10(temp) * signs 
#grad2[t] = np.power(grad2[t], 10e13)
#grad2[t] = grad2[t] * 10e6
print(grad2[t])
#plt.plot(xrans, data2[0], 'o')
clor = 'viridis'
gradient = np.linspace(0, 1, 39)
gradient = np.vstack((gradient, gradient))
#fig, axes = plt.subplots(1)
#axes.imshow(gradient, aspect='auto', cmap=plt.get_cmap(clor))
#pos = list(axes.get_position().bounds)
#x_text = pos[0] - 0.01
#y_text = pos[1] + pos[3]/2
#fig.text(x_text, y_text, clor, va='center', ha='right', fontsize=10)
#axes.set_axis_off()
#plt.scatter(xrans, data2[t], c=grad2[t], cmap=plt.get_cmap('viridis'), vmin=min(grad2[t]), vmax=max(grad2[t]))
#area = (30 * np.random.rand(101))**2
#print(area)
#plt.scatter(xrans, data2[t], c=grad2[t], cmap=plt.cm.gist_rainbow, s=40, alpha=0.5)
#plt.scatter(xrans, grad2[t], c=grad2[t], cmap=plt.cm.gist_rainbow, s=20 )
print(min(grad2[t]))
print(max(grad2[t]))
#plt.colorbar()
#plt.show()
grad2 = np.abs(normalize(grad2, norm="l2"))
ax = sns.heatmap(grad2, cmap="YlGnBu", cbar_kws={'label':' Jacobian '}, vmin=-1e-6,vmax=1e-6)
ax.set(ylabel='Features', xlabel='Timesteps')
#plt.colorbar( label='gradient') #ticks=range(100),
plt.show()
