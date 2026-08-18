import numpy as np
import matplotlib.pyplot as plt

# Linear Airy wave in finite depth.
# eta = a cos(kx - omega t)
# phi = (a*g/omega) * cosh(k(z+h))/cosh(kh) * sin(kx-omega*t)
g = 9.81
a = 0.05
h = 5.0
T = 3.0
omega = 2*np.pi/T

# solve dispersion relation omega^2 = g k tanh(kh)
from scipy.optimize import brentq
f = lambda k: g*k*np.tanh(k*h) - omega**2
k = brentq(f, 1e-6, 10.0)

x = np.linspace(-10, 10, 300)
z = np.linspace(-h, 0, 100)
X, Z = np.meshgrid(x, z)
t = 0.0

eta = a*np.cos(k*X - omega*t)
phi = (a*g/omega) * np.cosh(k*(Z+h))/np.cosh(k*h) * np.sin(k*X-omega*t)

np.savez("results/analytic_wave.npz", x=x, z=z, eta=eta, phi=phi, k=k, omega=omega)

plt.figure()
plt.plot(x, a*np.cos(k*x))
plt.xlabel("x [m]")
plt.ylabel("surface elevation [m]")
plt.title(f"Airy wave: T={T:.2f} s, h={h:.1f} m, k={k:.4f} 1/m")
plt.grid(True)
plt.savefig("results/analytic_surface.png", dpi=200, bbox_inches="tight")
plt.close()

print("k =", k)
print("wavelength =", 2*np.pi/k)
print("omega =", omega)
