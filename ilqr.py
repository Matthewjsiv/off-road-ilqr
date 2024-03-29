import numpy as np
import torch
from utils import *
import time
from utils import *
import matplotlib.pyplot as plt
import cv2

torch.manual_seed(0)

class ILQR(object):
    def __init__(self, model, Q, R, Qf, iterations = 50, tol = 1e-3):
        self.model = model
        self.iterations = iterations
        self.tol = tol
        self.obs= None
        self.Q = torch.FloatTensor(Q)
        self.R = torch.FloatTensor(R)
        self.Qf = torch.FloatTensor(Qf)
        self.Qc = torch.FloatTensor([.01])
        self.nx = 3
        self.nu = 2

    def run(self, Xref, Uref, observation):
        """
        for now - X - Nx3, U - (N-1)x2
        """

        now = time.perf_counter()

        # self.obs = observation
        self.costmap = observation['costmap']
        # self.costmap = torch.zeros_like(self.costmap)
        self.obs = observation['state']
        self.res = observation['res']

        Xi = Xref[0]
        U = Uref + torch.randn(Uref.shape)*.1

        X = self.model.rollout(Xi.unsqueeze(0), U.unsqueeze(0))
        X = X[0]

        
        #plt.show()

        for i in range(self.iterations):

            # now = time.perf_counter()
            d,K, delta_J = self.backward_pass(X, U, Xref, Uref)
            X,U, J = self.forward_pass(X,U,d,K, Xref, Uref)

            # plt.plot(Xref[:,0], Xref[:,1])
            # plt.plot(Xref[:,0], Xref[:,1])
            
            # plt.show()

            print(delta_J)

            if delta_J < self.tol:
                plt.show()
                print(time.perf_counter() - now)

                costmap = self.costmap
                odom = self.obs
                res = self.res
                pose_se3 = pose_msg_to_se3(odom)
                #we're just using this for rotation
                traj = transformed_lib(pose_se3, X)

                traj = ((traj - np.array([-30., -30])) / res)

                traj_ref = transformed_lib(pose_se3, Xref)

                traj_ref = ((traj_ref - np.array([-30., -30])) / res)


                plt.imshow(costmap,origin='lower', extent=[-30,30,-30,30])
                plt.plot(traj_ref[:,0], traj_ref[:,1])
                plt.plot(traj[:,0], traj[:,1])
                plt.show()

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

        P[-1] = dJdx2
        p[-1] = dJdx

        # print(dJdx2, dJdx)

        dJ = 0.0

        # print(X)
        self.costmap_expansion(X)

        for k in range(N-2,-1,-1):
            dJdx2, dJdx, dJdu2, dJdu = self.stage_cost_expansion(X[k].unsqueeze(0), U[k].unsqueeze(0), Xref[k].unsqueeze(0), Uref[k].unsqueeze(0), k)

            # print(dJdx2.shape, dJdx.shape)

            #TODO actually do this
            A, B = self.model.dynamics_jacobians(X[k], U[k])

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
    
    def forward_pass(self, X, U, d, K, Xref, Uref, tol = 1e-4):
        Xn = torch.zeros_like(X)
        Un = torch.zeros_like(U)

        Xn[0] = X[0]

        a = 1.0

        J = self.trajectory_cost(X, U, Xref, Uref)
        for i in range(25):
            for k in range(X.shape[0] - 1):
                Un[k] = U[k] - a*d[k].flatten() - K[k] @ (Xn[k]-X[k])
                Xn[k+1] = self.model.predict(Xn[k].unsqueeze(0), Un[k].unsqueeze(0))

            #print(torch.linalg.norm(Xn-X))

            Jn = self.trajectory_cost(Xn, Un, Xref, Uref)
            if Jn < J + tol:
                return Xn, Un, J
            
            a = 0.5 * a

        raise Exception("Line search didn't converge")
    
    def trajectory_cost(self, X, U, Xref, Uref):
        J = 0

        for k in range(X.shape[0] - 1):
            J += self.stage_cost(X[k], U[k], Xref[k], Uref[k])

        J += self.terminal_cost(X[k], Xref[k])

        return J

    def costmap_cost(self, x):
        """
        rn assuming one coordinate but we should vectorize it back later since not needed for backward pass
        """
        costmap = self.costmap
        odom = self.obs
        res = self.res
        pose_se3 = pose_msg_to_se3(odom)
        #we're just using this for rotation
        traj = transformed_lib(pose_se3, x.unsqueeze(0))

        xdisc = ((traj - np.array([-30., -30])) / res).long()

        cost = costmap[xdisc[:,0], xdisc[:,1]]

        return cost * self.Qc

    def stage_cost(self, x, u, xref, uref):
        # tracking_cost = .5*(np.sum((X-Xref)**2))*
        tracking_cost = .5*(x-xref).T @ self.Q @ (x-xref) + .5*(u-uref).T @ self.R @ (u-uref)

        costmap_cost = self.costmap_cost(x.clone())

        return tracking_cost + costmap_cost

    def terminal_cost(self, x, xref):
        # tracking_cost = .5*(np.sum((X-Xref)**2))*
        tracking_cost = .5*(x-xref).T @ self.Qf @ (x-xref)

        costmap_cost = self.costmap_cost(x.clone())

        return tracking_cost + costmap_cost

    def stage_cost_expansion(self, x, u, xref, uref, k):
        dtrackingdx = self.Q @ (x-xref).T
        #TODO actually do this
        dcostmapdx = torch.zeros([3,1])
        dcostmapdx[:2] = self.dcostmapdx[k].view(2,1)
        dJdx = dtrackingdx + dcostmapdx

        dtrackingdx2 = self.Q
        #TODO actually do this
        dcostmapdx2 = torch.zeros([3,3])
        # print(self.dcostmapdx2[k,:2].view(2,1).shape)
        dcostmapdx2[0,:2] = self.dcostmapdx2[k,:2].view(1,2)
        dcostmapdx2[1,:2] = self.dcostmapdx2[k,2:].view(1,2)
        dJdx2 = dtrackingdx2 + dcostmapdx2

        dJdu = self.R @ (u-uref).T
        dJdu2 = self.R

        return dJdx2, dJdx, dJdu2, dJdu

    def costmap_expansion(self, X):
        costmap = self.costmap
        odom = self.obs
        res = self.res
        pose_se3 = pose_msg_to_se3(odom)
        # print(odom)
        #we're just using this for rotation
        trajs = transformed_lib(pose_se3, X)
        # print(X)
        # print(trajs)
        trajs_disc = ((trajs - np.array([-30., -30])) / res).long()
        # print(trajs_disc[:5])

        # costmap[trajs_disc[:,0],trajs_disc[:,1]] = 1
        # # print(costmap.shape)
        # plt.imshow(costmap)
        # plt.show()
        # cv2.imshow('test', costmap.cpu().numpy())
        # cv2.waitKey(1)

        #TODO THESE MIGHT NEED TO BE FLIPPED
        dcx,dcy = torch.gradient(costmap)

        dcxx,dcxy = torch.gradient(dcx)

        dcyx,dcyy = torch.gradient(dcy)

        dcx_traj = dcx[trajs_disc[:,0], trajs_disc[:,1]]
        dcy_traj = dcy[trajs_disc[:,0], trajs_disc[:,1]]

        dcxx_traj = dcxx[trajs_disc[:,0], trajs_disc[:,1]]
        dcxy_traj = dcxy[trajs_disc[:,0], trajs_disc[:,1]]

        dcyy_traj = dcyy[trajs_disc[:,0], trajs_disc[:,1]]
        dcyx_traj = dcyx[trajs_disc[:,0], trajs_disc[:,1]]

        self.dcostmapdx = torch.stack([dcx_traj, dcy_traj]).T * self.Qc
        # print(self.dcostmapdx.shape)
        self.dcostmapdx2 = torch.stack([dcxx_traj, dcxy_traj, dcyx_traj, dcyy_traj]).T * self.Qc
        # print(self.dcostmapdx2.shape)




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
