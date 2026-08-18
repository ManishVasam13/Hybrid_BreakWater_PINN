# Hybrid Breakwater–Wave Energy Converter PINN

Educational/research MVP for a 4-month engineering project.

## Scientific scope
This implementation solves a 2-D linear potential-flow wave problem with a Physics-Informed Neural Network (PINN). The baseline field is the velocity potential `phi(x,z,t)` in a rectangular water domain. The model enforces:

- Laplace equation in the fluid: phi_xx + phi_zz = 0
- Linearized free-surface dynamic/kinematic conditions, combined as a second-order surface condition
- No-penetration at the seabed
- A simplified rigid vertical breakwater boundary
- A prescribed oscillating WEC boundary condition

The first implementation is intentionally simplified so that every part can be understood and validated. It is NOT a full Navier–Stokes/free-surface CFD model.

## What you will learn
1. Python project structure
2. PyTorch tensors and neural networks
3. Automatic differentiation
4. PDE residual construction
5. Boundary-condition losses
6. Collocation sampling
7. PINN training and diagnostics
8. Wave theory and linear potential flow
9. Hydrodynamic pressure/force from velocity potential
10. WEC/PTO power concepts
11. Validation against an analytical Airy-wave solution
12. Parameter studies and research reporting

## Setup

Create a virtual environment:

    python -m venv .venv
    .venv\Scripts\activate

Install:

    pip install -r requirements.txt

Run the validation/training example:

    python src/train.py

Then create plots:

    python src/postprocess.py

Outputs are written to `results/`.

## Recommended workflow
Do not jump directly to a large PINN. Run the scripts in this order:

1. `src/01_python_basics.py`
2. `src/02_autograd_demo.py`
3. `src/03_analytic_wave.py`
4. `src/train.py`
5. `src/postprocess.py`

## Research extensions
After the baseline works, add:
- Fourier features for wave-frequency representation
- adaptive collocation
- parameterized wave period/depth/amplitude
- breakwater permeability
- prescribed WEC motion
- coupled WEC dynamics
- sparse synthetic/experimental data loss
- comparison with finite-difference/FEM/CFD
