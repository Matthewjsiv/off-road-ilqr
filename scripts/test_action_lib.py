import torch
import numpy as np
import matplotlib.pyplot as plt
#temp hack for now
import sys
sys.path.append("../")
from models.kbm import KBM
actions = torch.load('../data/vel_lib.pt')
# print(actions.shape)

kbm = KBM(L=3.0)


Xi = torch.Tensor([0,0,np.pi/2])
X = kbm.rollout(Xi, actions)

for n in range(2):
    ids = np.random.choice(X.shape[0],50)
    Xsub = X[ids]

    for i in range(Xsub.shape[0]):
        plt.plot(Xsub[i,:,0], Xsub[i,:,1])

    plt.show()

print(X.shape)
print(actions.shape)
