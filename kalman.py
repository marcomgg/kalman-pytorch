import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


def jacobian(x, y):
    batch_size, x_dim = x.shape
    device = y.device
    _, y_dim = y.shape
    j = torch.zeros(batch_size, x_dim, y_dim).to(device)
    for i in range(y_dim):
        grad, = torch.autograd.grad(y[:, i], x, torch.ones(batch_size).to(device), create_graph=True)
        j[:, :, i] = grad
    return j


class System(nn.Module):
    def __init__(self, x_dim, y_dim, u_dim, lx, ly):
        super().__init__()
        self.x_dim, self.u_dim, self.y_dim = x_dim, u_dim, y_dim
        self.lx, self.ly = nn.Parameter(lx), nn.Parameter(ly)

    def f(self, x, u):
        raise NotImplementedError

    def h(self, x, u):
        raise NotImplementedError

    def linearize(self, x, u):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def _get_device(self):
        return self.lx.device

    def _get_sigmax(self):
        return self.lx.matmul(self.lx.transpose(1,0))

    def _get_sigmay(self):
        return self.ly.matmul(self.ly.transpose(1,0))

    device = property(_get_device)
    sigmax = property(_get_sigmax)
    sigmay = property(_get_sigmay)


class LinearSystem(System):

    def __init__(self, a, b, c, lx, ly):
        super().__init__(x_dim=a.shape[0], y_dim=c.shape[0], u_dim=b.shape[1], lx=lx, ly=ly)
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)

    def forward(self, x_0, u, n):
        x_k = x_0
        x = torch.zeros(x_0.shape[0], n, self.x_dim).to(self.device)
        y = torch.zeros(x_0.shape[0], n, self.y_dim).to(self.device)

        noise = torch.randn_like(x).view(-1, 1, n, self.x_dim)
        noisex = self.lx.matmul(noise.transpose(3, 2).to(self.device)).squeeze(dim=1).transpose(2, 1)

        for k in range(n):
            # State transition
            x_k = self.f(x_k, u[:, k, :]) + noisex[:, k, :]
            x[:, k, :] = x_k
            # Output transition
            y_k = self.h(x_k, None)
            y[:, k, :] = y_k

        # Add output noise
        noise = torch.randn_like(x).view(-1, 1, n, self.y_dim)
        noisey = self.ly.matmul(noise.transpose(3, 2).to(self.device)).squeeze(dim=1).transpose(2, 1)
        y = y + noisey

        return x, y

    def prediction(self, x_0, u, n):
        x_k = x_0
        x = torch.zeros(x_0.shape[0], n, self.x_dim).to(self.device)
        y = torch.zeros(x_0.shape[0], n, self.y_dim).to(self.device)

        for k in range(n):
            # State transition
            x_k = self.f(x_k, u[:, k, :])
            x[:, k, :] = x_k
            # Output transition
            y_k = self.h(x_k, None)
            y[:, k, :] = y_k
        return x, y

    def f(self, x, u):
        return x.matmul(self.a.transpose(1, 0)) + u.matmul(self.b.transpose(1, 0))

    def h(self, x, u):
        return x.matmul(self.c.transpose(1, 0))

    def linearize(self, x, u):
        return self.a[None, :, :], self.c[None, :, :]


class NonLinearSystem(System):
    '''
        Base class for a non linear system, derive this and implement
        the members forward, f and h using PyTorch function
    '''
    def linearize(self, x, u):
        x.requires_grad = True
        f = self.f(x, u)
        h = self.h(x, u)
        F = jacobian(x, f)
        H = jacobian(x, h)
        return F, H


class Kalman(nn.Module):
    """"
        Module that implements a Kalman filter, we follow the notation from
        M. Bishop "Pattern recognition and machine learning" chapter 13.3.1
    """
    def __init__(self, system):
        super().__init__()
        self.system = system

    def _get_device(self):
        return self.system.device

    device = property(_get_device)

    def forward(self, y):
        pass

    def filter(self, y, u, mu0, P0):
        s = self.system
        batch_size, n, _ = y.shape

        mu_sequence = torch.zeros(batch_size, n, s.x_dim).to(self.device)
        y_sequence = torch.zeros(batch_size, n, s.y_dim).to(self.device)
        vy_sequence = torch.zeros(batch_size, n, s.y_dim, s.y_dim).to(self.device)
        logprob = 0
        v_sequence = torch.zeros(batch_size, n, s.x_dim, s.x_dim).to(self.device)
        eye = torch.eye(s.x_dim).view(-1, 1, s.x_dim, s.x_dim).repeat(batch_size, 1, 1, 1).to(self.device)

        # Linearize the system
        F, H = s.linearize(mu0, u[:, 0, :])
        F = F.view(-1, 1, s.x_dim, s.x_dim)
        H = H.view(-1, 1, s.y_dim, s.x_dim)

        # Kalman gain at time 0
        ph = P0.view(-1, 1, s.x_dim, s.x_dim).matmul(H.transpose(3, 2))
        hph = H.matmul(ph)
        k = ph.matmul((hph + s.sigmay).double().inverse().float())

        # Residual at time 0
        ypred = s.h(mu0.view(-1, s.x_dim), None)
        delta = y[:, 0, :] - ypred

        # Posterior mean (filtered state) at time 0
        mu = mu0 + k.matmul(delta.view(-1, 1, s.y_dim, 1)).view(-1, s.x_dim)

        # Posterior covariance at time 0
        v = (eye - k.matmul(H)).matmul(P0.view(-1, 1, s.x_dim, s.x_dim))

        # Likelihood
        mu_sequence[:, 0, :] = mu
        v_sequence[:, 0, :, :] = v.squeeze(dim=1)
        y_sequence[:, 0, :] = ypred
        vy_sequence[:, 0, :, :] = (hph + s.sigmay).squeeze(dim=1)

        for i in range(1, n):
            yi = y[:, i, :]; ui = u[:, i, :]
            F, H = s.linearize(mu.view(-1, s.x_dim).data, ui)
            F = F.view(-1, 1, s.x_dim, s.x_dim)
            H = H.view(-1, 1, s.y_dim, s.x_dim)

            # Prior covariance
            p = F.matmul(v).matmul(F.transpose(3, 2)) + s.sigmax

            # Kalman gain
            ph = p.matmul(H.transpose(3, 2))
            hph = H.matmul(ph) + s.sigmay
            k = ph.matmul(hph.double().inverse().float())

            # Residual
            ypred = s.h(s.f(mu.view(-1, s.x_dim), ui), None)
            delta = yi - ypred

            # Posterior mean
            mu = s.f(mu.view(-1, s.x_dim), u[:, i, :]) + \
                k.matmul(delta.view(-1, 1, s.y_dim, 1)).view(-1, s.x_dim)

            # Posterior covariance
            v = (eye - k.matmul(H)).matmul(p.view(-1, 1, s.x_dim, s.x_dim))

            mu_sequence[:, i, :] = mu
            v_sequence[:, i, :, :] = v.squeeze(dim=1)
            y_sequence[:, i, :] = ypred
            vy_sequence[:, i, :, :] = hph.squeeze(dim=1)

        try:
            c = MultivariateNormal(y_sequence.view(-1, s.y_dim),
                                   vy_sequence.view(-1, s.y_dim, s.y_dim))
            logp = c.log_prob(y.view(-1, s.y_dim)).mean()
        except Exception:
            print(vy_sequence)
            exit(1)
        return mu_sequence, v_sequence, logp, y_sequence, vy_sequence

    def smoother(self, y, u, mu0, P0):
        s = self.system
        batch_size, n, _ = y.shape

        # Pre-compute the input
        ux = s.b.matmul(u.view(-1, 1, n, s.u_dim).transpose(3, 2)).squeeze(1).transpose(2, 1)

        mu_sequence = torch.zeros(batch_size, n, s.x_dim).to(self.device)
        v_sequence = torch.zeros(batch_size, n, s.x_dim, s.x_dim).to(self.device)

        mu_filter, v_filter, logp, y_filter, vy_filter = self.filter(y, u, mu0, P0)
        mu_smooth, v_smooth = mu_filter[:, n-1, :].view(-1, 1, s.x_dim, 1), v_filter[:, n-1, :, :].view(-1, 1, s.x_dim, s.x_dim)
        v_sequence[:, n-1, :, :] = v_smooth.squeeze(dim=1)
        mu_sequence[:, n-1, :] = mu_smooth.view(-1, s.x_dim)

        for i in range(n-2, -1, -1):
            mu, v = mu_filter[:, i, :].view(-1, 1, s.x_dim, 1), v_filter[:, i, :, :].view(-1, 1, s.x_dim, s.x_dim)

            # Prior covariance at t+1
            p = s.a.matmul(v).matmul(s.a.transpose(1, 0)) + s.sigmax

            # Smoother gain
            j = v.matmul(s.a.transpose(1, 0)).matmul(p.inverse())

            # Smoothed mean
            mu_smooth = mu + j.matmul(mu_smooth - s.a.matmul(mu.view(-1, 1, s.x_dim, 1)) - ux[:, i+1, :].view(-1, 1, s.x_dim, 1))

            # Smoothed covariance
            v_smooth = v + j.matmul(v_smooth - p).matmul(j.transpose(3, 2))
            mu_sequence[:, i, :] = mu_smooth.view(-1, s.x_dim)
            v_sequence[:, i, :, :] = v_smooth.squeeze(dim=1)

        return mu_sequence, v_sequence, mu_filter, v_filter, logp, y_filter, vy_filter
