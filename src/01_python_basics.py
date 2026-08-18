import numpy as np

# Basic tensors/arrays you need before PINNs
x = np.linspace(0, 10, 6)
print("x =", x)
print("x^2 =", x**2)

# A simple wave
A = 0.05
k = 2*np.pi/10
omega = 2.0
t = 0.0
eta = A*np.cos(k*x - omega*t)
print("wave elevation =", eta)
