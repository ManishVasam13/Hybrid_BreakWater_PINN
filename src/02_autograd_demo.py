import torch

# y = x^3 + 2x
x = torch.tensor([[2.0]], requires_grad=True)
y = x**3 + 2*x

dy_dx = torch.autograd.grad(
    y, x, grad_outputs=torch.ones_like(y), create_graph=True
)[0]

d2y_dx2 = torch.autograd.grad(
    dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True
)[0]

print("y =", y.item())
print("dy/dx =", dy_dx.item(), " expected =", 3*2**2 + 2)
print("d2y/dx2 =", d2y_dx2.item(), " expected =", 6*2)
