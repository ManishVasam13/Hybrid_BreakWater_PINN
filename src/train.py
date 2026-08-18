import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pinn import MLP, laplace, surface_operator, grad

torch.manual_seed(7)
np.random.seed(7)

os.makedirs("results", exist_ok=True)

# Physical parameters
g = 9.81
h = 5.0
a = 0.05
T = 3.0
omega = 2*np.pi/T

# Domain: x in [-10,10], z in [-h,0], t in [0,T]
xmin, xmax = -10.0, 10.0
zmin, zmax = -h, 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)

# Normalize inputs to roughly [-1,1]
def make_X(x,z,t,requires_grad=True):
    X = torch.tensor(np.column_stack([x,z,t]), dtype=torch.float32, device=device)
    X[:,0] = 2*(X[:,0]-xmin)/(xmax-xmin)-1
    X[:,1] = 2*(X[:,1]-zmin)/(zmax-zmin)-1
    X[:,2] = 2*(X[:,2]/T)-1
    X.requires_grad_(requires_grad)
    return X

# Airy-wave dispersion relation for the inlet boundary.
from scipy.optimize import brentq
k = brentq(lambda kk: g*kk*np.tanh(kk*h)-omega**2, 1e-6, 10.0)

def phi_airy(x,z,t):
    return (a*g/omega)*np.cosh(k*(z+h))/np.cosh(k*h)*np.sin(k*x-omega*t)

# Collocation points
Nf = 12000
Nb = 2000
Ns = 2000

def uniform(n):
    return np.random.rand(n)

# Interior PDE points
x = xmin + (xmax-xmin)*uniform(Nf)
z = zmin + (zmax-zmin)*uniform(Nf)
t = T*uniform(Nf)
Xf = make_X(x,z,t)

# Free surface
xs = xmin + (xmax-xmin)*uniform(Ns)
ts = T*uniform(Ns)
Xs = make_X(xs, np.zeros(Ns), ts)

# Seabed
xb = xmin + (xmax-xmin)*uniform(Nb)
tb = T*uniform(Nb)
Xb = make_X(xb, np.full(Nb,zmin), tb)

# Inlet: impose analytical incoming Airy potential.
xi = np.full(Ns, xmin)
ti = T*uniform(Ns)
zi = zmin + (zmax-zmin)*uniform(Ns)
Xi = make_X(xi, zi, ti)

mse = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)

history = []

for epoch in range(1, 5001):
    opt.zero_grad()

    phif = model(Xf)
    r_pde = laplace(phif, Xf)
    loss_pde = torch.mean(r_pde**2)

    phis = model(Xs)
    r_fs = surface_operator(phis, Xs, omega, g)
    loss_fs = torch.mean(r_fs**2)

    phib = model(Xb)
    phi_z_b = grad(phib, Xb, 1)
    loss_b = torch.mean(phi_z_b**2)

    phii = model(Xi)
    target_i = torch.tensor(
        phi_airy(xi, zi, ti), dtype=torch.float32, device=device
    ).view(-1,1)
    loss_i = mse(phii, target_i)

    loss = loss_pde + 5.0*loss_fs + loss_b + 10.0*loss_i
    loss.backward()
    opt.step()

    history.append([epoch, loss.item(), loss_pde.item(), loss_fs.item(), loss_b.item(), loss_i.item()])

    if epoch % 250 == 0:
        print(
            f"{epoch:5d} total={loss.item():.3e} "
            f"pde={loss_pde.item():.3e} fs={loss_fs.item():.3e} "
            f"bed={loss_b.item():.3e} inlet={loss_i.item():.3e}"
        )

np.savetxt("results/loss_history.csv", np.array(history), delimiter=",",
           header="epoch,total,pde,free_surface,bed,inlet", comments="")
torch.save(model.state_dict(), "results/pinn_model.pt")

# Evaluate field at t=0
nx, nz = 250, 100
xx = np.linspace(xmin,xmax,nx)
zz = np.linspace(zmin,zmax,nz)
XX, ZZ = np.meshgrid(xx,zz)
tt = np.zeros_like(XX)

XE = make_X(XX.ravel(), ZZ.ravel(), tt.ravel(), requires_grad=False)
with torch.no_grad():
    pred = model(XE).cpu().numpy().reshape(nz,nx)

np.savez("results/pinn_field.npz", x=xx,z=zz,phi=pred,k=k,omega=omega,h=h)

plt.figure(figsize=(10,4))
plt.pcolormesh(XX, ZZ, pred, shading="auto")
plt.colorbar(label="PINN velocity potential")
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.title("PINN velocity potential at t=0")
plt.savefig("results/pinn_phi.png", dpi=200, bbox_inches="tight")
plt.close()
