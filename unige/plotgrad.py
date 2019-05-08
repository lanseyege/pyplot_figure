import numpy as np
import pickle
import matplotlib.pyplot as plt

path = './save/CV_10_ones.3grad'
grad = pickle.load(open(path, 'rb'))

path = './save/CV_10_ones.3data'
data = pickle.load(open(path, 'rb'))
print(data)
(n, m) = data.shape

def sigm(x):
    return 1.0/(1.0 + np.exp(-x)) - 0.5

t = 20
xrans = np.arange(n)
data2 = np.transpose(data)
grad2 = np.transpose(grad)
signs = np.sign(grad2)
temp = np.abs(grad2)
#grad2 = -0.0001/np.log10(temp) * signs 
#grad2[t] = np.power(grad2[t], 10e13)
#grad2[t] = grad2[t] * 10e6
print(grad2[t])
#plt.plot(xrans, data2[0], 'o')

clor = 'viridis'
lmin, lmax = [], []
anames = ['RKneeAngles', 'LThoraxAngles', 'LHipAngles', 'RThoraxAngles', 'LFootProgressAngles', 'RHipAngles',  'LKneeAngles', 'RSpineAngles', 'RFootProgressAngles', 'RPelvisAngles',  'LAnkleAngles', 'LPelvisAngles', 'RAnkleAngles']

for i in range(39):
    #fig, axes = plt.subplots(1)
    plt.scatter(xrans, data2[i], c=grad2[i], cmap=plt.cm.gist_rainbow, s=40,alpha=0.5)
    #plt.scatter(xrans, grad2[t], c=grad2[t], cmap=plt.cm.gist_rainbow, s=20 )
    print(min(grad2[i]))
    print(max(grad2[i]))
    lmin.append(min(grad2[i]))
    lmax.append(max(grad2[i]))
    nm = anames[int(i/13)] + '_' + str(i%13 + 1)
    plt.xlabel('Timesteps')
    plt.ylabel(nm)
    plt.colorbar( label='Jacobian') #ticks=range(100),
    #plt.clim(-1e-5, 1e-5)
    #plt.show()

    plt.savefig('./save/apic/pic_rnn/'+str(i+1)+'.png')
    plt.close()
#print(min(lmin))
#print(max(lmax))
