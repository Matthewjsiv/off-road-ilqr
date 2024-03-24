import numpy as np
import torch
from utils import *


class ILQR(object):
    def __init__(self, model, Q, R, Qf, iterations = 500, tol = 1e-3):
        self.model = model
        self.iterations = iterations
        self.tol = tol
        self.obs= None
        self.Q = torch.FloatTensor(Q)
        self.R = torch.FloatTensor(R)
        self.Qf = torch.FloatTensor(Qf)
        self.nx = 3
        self.nu = 2

    def run(self, Xref, Uref, observation):
        """
        for now - X - Nx3, U - (N-1)x2
        """

        self.obs = observation

        Xi = Xref[0]
        U = Uref

        #X
        X = self.model.rollout(Xi.unsqueeze(0), U.unsqueeze(0))
        X = X[0]

        for i in range(self.iterations):

            d,K, delta_J = self.backward_pass(X, U, Xref, Uref)
            X,u, J = self.forward_pass(X,U,d,K)

            if delta_J < self.tol:
                return X, U, K

        print("FAILED TO CONVERGE :o")
        return None

    def backward_pass(self, X, U, Xref, Uref):

        N = X.shape[0]

        P = [torch.zeros([self.nx, self.nx])] * N
        p = [torch.zeros([self.nx,1])] * N
        d = [torch.zeros([self.nu,1])] * (N-1)
        K = [torch.zeros([self.nu,self.nx])] * (N-1)

        dJdx2, dJdx = self.terminal_cost_expansion(X[-1].unsqueeze(0), Xref[-1].unsqueeze(0))
        dJ = 0.0

        for k in range(N-2,-1,-1):
            dJdx2, dJdx, dJdu2, dJdu = self.stage_cost_expansion(X[k].unsqueeze(0), U[k].unsqueeze(0), Xref[k].unsqueeze(0), Uref[k].unsqueeze(0))

            #TODO actually do this
            A = torch.zeros([3,3])
            B = torch.zeros([3,2])

            # print(dJdx.dtype, A.dtype, p[0].dtype)
            gx = dJdx + A.T @ p[k+1]

            gu = dJdu + B.T @ p[k+1]

            Gxx = dJdx2 + A.T @ P[k+1] @ A
            Guu = dJdu2 + B.T @ P[k+1] @ B
            Gxu = A.T @ P[k+1] @ B
            Gux = B.T @ P[k+1] @ A

            #TODO use pinv here??
            d[k] = torch.linalg.pinv(Guu) @ gu
            K[k] = torch.linalg.pinv(Guu) @ Gux

            # print(gx.shape, K[k].T.shape, gu.shape)
            # import pdb;pdb.set_trace()
            p[k] = gx - K[k].T @ gu + K[k].T @ Guu @ d[k] - Gxu @ d[k]
            P[k] = Gxx + K[k].T @ Guu @ K[k] - Gxu @ K[k] - K[k].T @ Gux

            dJ += gu.T @ d[k]

        return d, K, dJ



    def costmap_cost(x):
        """
        rn assuming one coordinate but we should vectorize it back later since not needed for backward pass
        """
        costmap = self.obs['local_costmap_data'][t][0].numpy()
        odom = self.obs['state'][t]
        res = self.obs['local_costmap_resolution'][t][0].item() #assuming square
        pose_se3 = pose_msg_to_se3(odom)
        #we're just using this for rotation
        x[:,:2] = transformed_lib(pose_se3, np.expand_dims(x,0))

        xdisc = ((x[:,:2] - np.array([-30., -30]).reshape(1, 2)) / res).int()

        cost = costmap[xdisc[0,0], xdisc[0,1]]

        return cost

    def stage_cost(self, x, u, xref, uref):
        # tracking_cost = .5*(np.sum((X-Xref)**2))*
        tracking_cost = .5*(x-xref).T @ self.Q @ (x-xref) + .5*(u-uref).T @ self.R @ (u-uref)

        costmap_cost = self.costmap_cost(x.copy())

        return tracking_cost + costmap_cost

    def terminal_cost(self, x, xref):
        # tracking_cost = .5*(np.sum((X-Xref)**2))*
        tracking_cost = .5*(x-xref).T @ self.Qf @ (u-uref)

        costmap_cost = self.costmap_cost(x.copy())

        return tracking_cost + costmap_cost

    def stage_cost_expansion(self, x, u, xref, uref):
        dtrackingdx = self.Q @ (x-xref).T
        #TODO actually do this
        dcostmapdx = torch.zeros([3,1])
        dJdx = dtrackingdx + dcostmapdx

        dtrackingdx2 = self.Q
        #TODO actually do this
        dcostmapdx2 = torch.zeros([3,3])
        dJdx2 = dtrackingdx2 + dcostmapdx2

        dJdu = self.R @ (u-uref).T
        dJdu2 = self.R

        return dJdx2, dJdx, dJdu2, dJdu

    def terminal_cost_expansion(self, x, xref):
        #TODO CONFIRM ORDER
        dtrackingdx = self.Qf @ (x-xref).T
        #TODO actually do this
        dcostmapdx = torch.zeros([3,1])
        dJdx = dtrackingdx + dcostmapdx

        dtrackingdx2 = self.Qf
        #TODO actually do this
        dcostmapdx2 = torch.zeros([3,3])
        dJdx2 = dtrackingdx2 + dcostmapdx2

        return dJdx2, dJdx
