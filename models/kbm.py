import torch
import numpy as np
import gym

from torch_mpc.models.base import Model

class KBM(Model):
    """
    Kinematic bicycle model
    x = [x, y, th]
    u = [v, delta]
    xdot = [
        v * cos(th)
        v * sin(th)
        L / tan(delta)
    ]
    """
    def __init__(self, L=3.0, min_throttle=0., max_throttle=1., max_steer=0.3, dt=0.1, device='cpu'):
        self.L = L
        self.dt = dt
        self.device = device
        self.u_ub = np.array([max_throttle, max_steer])
        self.u_lb = np.array([min_throttle, -max_steer])
        self.max_steer = max_steer

    def observation_space(self):
        low = -np.ones(3).astype(float) * float('inf')
        high = -low
        return gym.spaces.Box(low=low, high=high)

    def action_space(self):
        return gym.spaces.Box(low=self.u_lb, high=self.u_ub)

    def dynamics(self, state, action):
        x, y, th = state.moveaxis(-1, 0)
        v, d = action.moveaxis(-1, 0)
        xd = v * th.cos()
        yd = v * th.sin()
        thd = v * d.tan() / self.L
        return torch.stack([xd, yd, thd], dim=-1)

    def dynamics_jacobians(self, state, action):
        x, y, th = state
        v, d = action

        A = torch.zeros(3,3)
        A[0, 2] = -v * torch.sin(th)
        A[1, 2] = v * torch.cos(th)

        B = torch.zeros(3,2)
        B[0, 0] = torch.cos(th)
        B[0, 0] = torch.sin(th)
        B[0, 1] = torch.tan(d) / self.L
        B[2, 1] = v * (torch.cos(d) ** -2) / self.L

        return A, B

    def predict(self, state, action):
        # print(action)
        # print(action.shape)
        if torch.abs(action[0,1]) > self.max_steer:
            # print("EXCEEDING CONTROLS")
            ...

        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + (self.dt/2)*k1, action)
        k3 = self.dynamics(state + (self.dt/2)*k2, action)
        k4 = self.dynamics(state + self.dt*k3, action)

        next_state = state + (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state

    def rollout(self, state, actions, append_cur_state = True):
        """
        Expected shapes:
            state: [B1 x ... x Bn x xd]
            actions: [B1 x ... x Bn x T x ud]
            returns: [B1 x ... x Bn X T x xd]
        """
        X = []
        curr_state = state
        if append_cur_state:
            X.append(state.repeat(actions.shape[0],1))
        for t in range(actions.shape[-2]):
            action = actions[..., t, :]
            next_state = self.predict(curr_state, action)
            X.append(next_state)
            curr_state = next_state.clone()

        return torch.stack(X, dim=-2)

    def quat_to_yaw(self, q):
        #quats are x,y,z,w
        qx, qy, qz, qw = q.moveaxis(-1, 0)
        return torch.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))

    def get_observations(self, batch):
        state = batch['state']
        if len(state.shape) == 1:
            return self.get_observations(dict_map(batch, lambda x:x.unsqueeze(0))).squeeze()

        x = state[..., 0]
        y = state[..., 1]
        q = state[..., 3:7]
        yaw = self.quat_to_yaw(q)
        return torch.stack([x, y, yaw], axis=-1)

    def get_actions(self, batch):
        return batch

    def get_speed(self, states):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        return self
