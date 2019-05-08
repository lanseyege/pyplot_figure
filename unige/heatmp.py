import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
from sklearn.preprocessing import normalize

path = './save/CV_10xg.3p'
#path = './save/CV_10cnn_xg.p'
#grad = pickle.load(open(path, 'rb'))
grad = torch.load(path).data.cpu().numpy()

t = 5
print(grad.shape)
(l, n, m) = grad.shape
grad  = grad[t]
xrans = np.arange(m-1)
grad2 = np.transpose(grad)

print(grad2[t])
#plt.plot(xrans, data2[0], 'o')
clor = 'viridis'
#print(min(grad2[t]))
#print(max(grad2[t]))
#plt.colorbar()
#plt.show()
grad2 = grad2[:, :-1]
grad2 = grad2.reshape(1,-1)
grad2 = np.abs(normalize(grad2, norm="l2"))
grad2 = grad2.reshape(-1, 100)
print(grad2)
print(grad2.shape)
#sys.exit()
plt.figure(figsize=(18.5, 10.5))
anames = ['RKneeAngles', 'LThoraxAngles', 'LHipAngles', 'RThoraxAngles', 'LFootProgressAngles', 'RHipAngles',  'LKneeAngles', 'RSpineAngles', 'RFootProgressAngles', 'RPelvisAngles',  'LAnkleAngles', 'LPelvisAngles', 'RAnkleAngles']

anames = ['LHipAngles', 'RSpineAngles', 'RPelvisAngles', 'LAnkleAngles', 'RKneeAngles', 'LFootProgressAngles', 'RAnkleAngles', 'RHipAngles', 'RFootProgressAngles', 'LThoraxAngles', 'LKneeAngles', 'RThoraxAngles', 'LPelvisAngles']
ylabels = [nm + '_' + str(i+1) for nm in anames for i in range(3)]
print(ylabels)
#ax = sns.heatmap(grad2, cmap="YlGnBu", cbar_kws={'label':' Jacobian '}, vmin=-1e-5, vmax=1e-5)
ax = sns.heatmap(grad2, cmap="YlGnBu", cbar_kws={'label':' Jacobian '}, )
ax.set(ylabel='Angles', xlabel='Timesteps')
#plt.colorbar( label='gradient') #ticks=range(100),
plt.yticks(np.arange(39), ylabels, rotation=0)
#plt.show()
plt.savefig('./save/apic/heat/norm_heat_rnn_newpatient_.pdf', format='pdf', dpi=1200, transparent=True, bbox_inches='tight', pad_inches=0.05)

plt.close()
