import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import torch 
from sklearn.preprocessing import normalize
from numpy.linalg import norm

names=['angles_aligned_test_.npy','angles_aligned_train_.npy','y_test.csv','y_tr    ain.csv']
#path = './save/CV_10_ones.3grad'
path = './save/CV_10xg.3p'
#path = './save/CV_10cnn_xg.p'
#grad = pickle.load(open(path, 'rb'))
grad = torch.load(path).data.cpu().numpy()

t = 0
print(grad.shape)
grad = grad[t]
path = './save/CV_10_ones.3data'
path = '../data/temp/CV_data/CV_10/angles_aligned/Left/'+names[1]
#path = '../data/CP/CV_data/CV_10/angles_aligned/Left/'+names[1]
data = np.load(path)
data = data.reshape(data.shape[0], data.shape[1], -1)
print(data.shape)
data = data[t]
(n, m) = data.shape

#sys.exit()
def sigm(x):
    return 1.0/(1.0 + np.exp(-x)) - 0.5

xrans = np.arange(n-1)
data2 = np.transpose(data)
grad2 = np.transpose(grad)
signs = np.sign(grad2)
temp = np.abs(grad2)
#grad2 = -0.0001/np.log10(temp) * signs 
#grad2[t] = np.power(grad2[t], 10e13)
#grad2[t] = grad2[t] * 10e6
#print(grad2[t])
#plt.plot(xrans, data2[0], 'o')

clor = 'viridis'
lmin, lmax = [], []
anames = ['RKneeAngles', 'LThoraxAngles', 'LHipAngles', 'RThoraxAngles', 'LFootProgressAngles', 'RHipAngles',  'LKneeAngles', 'RSpineAngles', 'RFootProgressAngles', 'RPelvisAngles',  'LAnkleAngles', 'LPelvisAngles', 'RAnkleAngles']
anames = ['LHipAngles', 'RSpineAngles', 'RPelvisAngles', 'LAnkleAngles', 'RKneeAngles', 'LFootProgressAngles', 'RAnkleAngles', 'RHipAngles', 'RFootProgressAngles', 'LThoraxAngles', 'LKneeAngles', 'RThoraxAngles', 'LPelvisAngles']
#for i in range(39):
#    gd = grad2[i]
#    normalize(gd, norm="l2")
#    print(gd)

#print(grad2)
grad2 = grad2[:,:-1]
print(grad2.shape)

#grad2 = np.abs(normalize(grad2, norm="l2", axis=0))
grad2 = grad2.reshape(1,-1)
grad2 = np.abs(normalize(grad2, norm="l2"))
grad2 = grad2.reshape(-1, 100)
print(grad2.shape)
#sys.exit()
#grad3, nm = normalize(grad2, norm="l2", axis=0, return_norm=True)
#print(nm)
#print(grad2)
#print(grad3)
#print(np.sum(grad3[0]))
#grad4 = grad2[0]/norm(grad2[0])
#print(np.sum(grad4))
#print(grad4)
#print(grad2)
#print(grad2[0])
#print(np.sum(grad2[:,0]))
#res = grad3/grad2
#print(grad3)
#print(np.sum(np.abs(grad2)/grad3))
#sys.exit()
for i in range(39):
    #grad4 = np.abs(grad2[i]/norm(grad2[i]))
    #fig, axes = plt.subplots(1)
    #plt.figure(figsize=(18.5, 10.5))
    plt.figure(figsize=(9.25, 5.25))
    plt.scatter(xrans, data2[i][:-1], c=grad2[i], cmap=plt.cm.gist_rainbow, s=40,alpha=0.5)
    #plt.scatter(xrans, grad2[t], c=grad2[t], cmap=plt.cm.gist_rainbow, s=20 )
    print(min(grad2[i]))
    print(max(grad2[i]))
    lmin.append(min(grad2[i]))
    lmax.append(max(grad2[i]))
    nm = anames[int(i/3)] + '_' + str(i%3 + 1)
    plt.xlabel('Timesteps')
    plt.ylabel(nm)
    plt.colorbar( label='Jacobian') #ticks=range(100),
    #plt.clim(-1e-6, 1e-6)
    #plt.show()
    plt.savefig('./save/apic5/0rnn/'+nm+'.pdf', format='pdf', dpi=1200, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close()
#print(min(lmin))
#print(max(lmax))

