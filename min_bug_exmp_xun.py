import math
import random
import tqdm
import torch
import torch.nn as nn
import numpy as np

seed=38383
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from torchdiffeq import odeint_adjoint as odeint
def logStdNorm(z):
    dim = z.size(-1)
    logZ = -0.5 * dim * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


class E(nn.Module):

    def __init__(self, cdim=128):
        super(E, self).__init__()
        self.fc = nn.Linear(3, cdim)

    def forward(self, x):
        return self.fc(x).mean(dim=1)


class F(nn.Module):

    def __init__(self, cdim=128):
        super(F, self).__init__()
        self.fc = nn.Linear(cdim, cdim)

    def forward(self, t, x):
        return self.fc(x)


class FCond(nn.Module):

    def __init__(self, cdim=128):
        super(FCond, self).__init__()
        self.fc = nn.Linear(3, 3)
        self.fc_scale = nn.Linear(cdim, 3)
        self.fc_bias  = nn.Linear(cdim, 3)

        self.fc_1 = nn.Linear(3, 3)
        self.fc_scale_1 = nn.Linear(cdim, 3)
        self.fc_bias_1  = nn.Linear(cdim, 3)

        self.fc_2 = nn.Linear(3, 3)
        self.fc_scale_2 = nn.Linear(cdim, 3)
        self.fc_bias_2  = nn.Linear(cdim, 3)

    def _forward_(self, x, c, fc, fc_bias, fc_scale):
        B, D = c.size(0), x.size(2)
        scale = torch.sigmoid(fc_scale(c).view(B, 1, D))
        bias = fc_bias(c).view(B, 1, D)
        return fc(x) * scale + bias


    def forward(self, t, x, c):
        x = self._forward_(x, c, self.fc, self.fc_bias, self.fc_scale)
        x = torch.tanh(x)

        x = self._forward_(x, c, self.fc_1, self.fc_bias_1, self.fc_scale_1)
        x = torch.tanh(x)

        x = self._forward_(x, c, self.fc_2, self.fc_bias_2, self.fc_scale_2)
        x = torch.tanh(x)

        x = self._forward_(x, c, self.fc_2, self.fc_bias_2, self.fc_scale_2)
        x = torch.tanh(x)

        x = self._forward_(x, c, self.fc_2, self.fc_bias_2, self.fc_scale_2)
        x = torch.tanh(x)

        x = self._forward_(x, c, self.fc_2, self.fc_bias_2, self.fc_scale_2)
        x = torch.tanh(x)

        return x


num_eval = 0

def divergence(f, y, e):
    global num_eval
    e_dfdy = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dfdy_e = e * e_dfdy
    out_shape = list(y.shape[:-1]) + [-1]
    approx_tr = e_dfdy_e.view(out_shape).sum(dim=-1)
    # if not approx_tr.requires_grad:
    #     print(approx_tr.requires_grad)
    # #     assert approx_tr.requires_grad
    # print(approx_tr.requires_grad, num_eval)
    # num_eval += 1
    return approx_tr


class ODENetCond(nn.Module):

    def __init__(self, odef):
        super(ODENetCond, self).__init__()
        self.odef = odef
        self.e_ = None

    def forward(self, t, states):
        global num_eval
        x, logp, c = states[0], states[1], states[2]
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            c.requires_grad_(True)
            t.requires_grad_(True)
            ret = self.odef(t, x, c)
            if self.e_ is None:
                self.e_ = torch.randn_like(ret).to(x)
            approx_tr = divergence(ret, x, self.e_).view(logp.size())
            print(t.item(), t.device, approx_tr.requires_grad, num_eval)
            num_eval += 1
                # print(approx_tr.requires_grad)
                # print(approx_tr.requires_grad)
            # assert approx_tr.requires_grad, "approx_tr.requires_grad should be True!"
        return [ret, -approx_tr, torch.zeros_like(c, requires_grad=True)]


class ODENet(nn.Module):

    def __init__(self, odef):
        super(ODENet, self).__init__()
        self.odef = odef
        self.e_ = None

    def forward(self, t, states):
        global num_eval
        x, logp = states[0], states[1]
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)
            ret = self.odef(t, x)
            if self.e_ is None:
                self.e_ = torch.randn_like(ret).to(x)
            approx_tr = divergence(ret, x, self.e_).view(logp.size())
            # print(approx_tr.requires_grad, num_eval)
            # num_eval += 1
        return [ret, -approx_tr]


class Trainer(nn.Module):

    def __init__(self):
        super(Trainer, self).__init__()

        self.enc = E()
        # ODE1
        self.f1 = F()
        self.ode1 = ODENet(self.f1)
        self.t1 = torch.tensor([0., 1.])

        # ODE2
        self.f2 = FCond()
        self.ode2 = ODENetCond(self.f2)
        self.t2 = torch.tensor([0., 1.])


    def forward(self, inp):
        global num_eval

        c = self.enc(inp)
        logpc = torch.zeros(list(c.size()[:-1])+[1]).to(c)
        y1, lp1 = odeint(self.ode1, (c, logpc), self.t1)

        num_eval = 0
        logpx = torch.zeros(list(inp.size()[:-1])+[1]).to(inp)
        y2, lp2, _ = odeint(self.ode2, (inp, logpx, c), self.t2)
        num_eval = 0

        # loss = (logStdNorm(y1) + lp1).mean()
               # + (logStdNorm(y2) + lp2).mean()
        loss = (logStdNorm(y2) + lp2).mean()
        return loss

model = Trainer().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
for _ in tqdm.trange(1000):
    inp = torch.rand(5,7,3).cuda()
    loss = model(inp)
    loss.backward()
    optimizer.step()


