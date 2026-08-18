import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden=64, depth=6):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def grad(y, x, dim):
    g = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return g[:, dim:dim+1]

def second_grad(y, x, dim):
    first = grad(y, x, dim)
    return grad(first, x, dim)

def laplace(phi, X):
    return second_grad(phi, X, 0) + second_grad(phi, X, 1)

def surface_operator(phi, X, omega, g):
    # Linearized combined free-surface condition at z=0:
    # phi_tt + g phi_z = 0
    phi_t = grad(phi, X, 2)
    phi_tt = grad(phi_t, X, 2)
    phi_z = grad(phi, X, 1)
    return phi_tt + g * phi_z
