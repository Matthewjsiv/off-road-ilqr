import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import cv2
import time
import sys
sys.path.append("../")
from models.kbm import KBM

# from ilqr import ILQR
from ilqr_exp_backup import ILQR
# from ilqr_numba import ILQR


class TrajLib_offline(object):
    def __init__(self, dataset_file):
        self.dataset = torch.load(dataset_file)
        print("dataset loaded")
        self.obs = self.dataset['observation']

    #preprocesses data at a given time t and returns cost and trajectories
    def preprocess_data(self, t, trajs):
        costmap = self.obs['local_costmap_data'][t][0].numpy()
        odom = self.obs['state'][t]
        res = self.obs['local_costmap_resolution'][t][0].item() #assuming square
        pose_se3 = pose_msg_to_se3(odom)
        #we're just using this for rotation
        trajs[:,:,:2] = transformed_lib(pose_se3, trajs)

        trajs_disc = ((trajs[:,:,:2] - np.array([-30., -30]).reshape(1, 1, 2)) / res).int()

        return trajs, trajs_disc, costmap



actions = torch.load('../data/vel_lib.pt')
# print(actions.shape)

kbm = KBM(L=3.0, max_steer=.3)

# traj_lib = TrajLib_offline(dataset_file='../data/context_mppi_pipe_1.pt')

actions[:,:,1] = 0.

Xi = torch.Tensor([0,0,0])
X = kbm.rollout(Xi, actions)

# choice = 457
choice = 720
Xref = X[choice]
Uref = actions[choice]
# print(Xref.shape, Uref.shape)
# plt.plot(Xref[:,0], Xref[:,1])
# plt.show()
# s=rs


all_obs = torch.load('../data/context_mppi_pipe_1.pt')['observation']

# for i in range(500,900,50):
#     # idx = 50
#     idx = i
#     # print(idx)
#     costmap = all_obs['local_costmap_data'][idx][0]
#
#     observation = {'costmap': costmap, 'state': all_obs['state'][idx],'res': all_obs['local_costmap_resolution'][idx][0]}
#
#     ilqr = ILQR(kbm, np.eye(3), np.eye(2), np.eye(3))
#     ilqr.run(Xref, Uref,observation)
#
#     break

idx = 50

costmap = all_obs['local_costmap_data'][idx][0]
# print(costmap.shape)
# costmap = torch.zeros(120,120).float()
# costmap[40:65,80:120] = 1.

costmap = np.zeros([120,120])
costmap = cv2.circle(costmap, (72,64), 2,1.0, -1)
costmap = cv2.circle(costmap, (85,57), 2,1.0, -1)
# costmap = cv2.circle(costmap, (70,70), 5,1.0, -1)

costmap = torch.Tensor(costmap).float()

observation = {'costmap': costmap, 'state': all_obs['state'][idx],'res': all_obs['local_costmap_resolution'][idx][0]}

ilqr = ILQR(kbm, np.eye(3), np.eye(2), np.eye(3))
ilqr.run(Xref, Uref,observation)
