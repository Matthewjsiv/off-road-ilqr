import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_mpc.models.kbm import KBM
from utils import *
import cv2
import time

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

kbm = KBM(L=3.0)

traj_lib = TrajLib_offline(dataset_file='../data/context_mppi_pipe_1.pt')

Xi = torch.Tensor([0,0,0])
X = kbm.rollout(Xi, actions)

for n in range(1000):
    # ids = np.random.choice(X.shape[0],50)
    # Xsub = X[ids]
    Xsub = X[::30].clone()

    Xsub,Xdisc, costmap = traj_lib.preprocess_data(n*50, Xsub)


    # costmap -= costmap.min()
    # costmap /= costmap.max()

    # for i in range(Xsub.shape[0]):
    #     try:
    #         costmap[Xdisc[i,:,0].int(), Xdisc[i,:,1].int()] = 1.
    #     except:
    #         ...

    # cv2.imshow('test', costmap)
    # cv2.waitKey(1)
    # time.sleep(.05)


    # fsize = 1
    # kernel = np.ones((fsize,fsize),np.float32)/(fsize**2)
    # costmap_blur = cv2.filter2D(costmap,-1,kernel)
    #
    # cx,cy = np.gradient(costmap_blur)
    # print(cx.max(), cx.min())
    #
    # fig, ax = plt.subplots(1,4)
    # ax[0].imshow(costmap)
    # ax[1].imshow(costmap_blur)
    # ax[2].imshow(cx)
    # ax[3].imshow(cy)
    #
    # plt.show()

    fsize = 15
    kernel = np.ones((fsize,fsize),np.float32)/(fsize**2)

    cx,cy = np.gradient(costmap)
    # print(cx.max(), cx.min())
    cx = cv2.filter2D(cx,-1,kernel)
    cy = cv2.filter2D(cy,-1,kernel)

    cx,cy = np.gradient(cx)
    # print(cx.max(), cx.min())
    cx = cv2.filter2D(cx,-1,kernel)
    cy = cv2.filter2D(cy,-1,kernel)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(costmap)
    ax[1].imshow(cx)
    ax[2].imshow(cy)

    plt.show()

    # plt.imshow(costmap,origin='lower',extent=[-30, 30, -30, 30])
    #
    # for i in range(Xsub.shape[0]):
    #     plt.plot(Xsub[i,:,0], Xsub[i,:,1])
    #
    # plt.show()

print(X.shape)
