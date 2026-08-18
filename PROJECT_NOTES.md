# Project notebook / report skeleton

## 1. Problem statement
Hydrodynamic analysis of a simplified hybrid breakwater–WEC using a PINN.

## 2. Research question
Can a PINN reproduce the velocity-potential field of linear surface waves while satisfying the governing equations and boundary conditions, and can the framework be extended to quantify the hydrodynamic environment around a hybrid breakwater–WEC?

## 3. Governing physics
Incompressible, inviscid, irrotational flow:

    ∇²φ = 0

Linearized free surface:

    φ_tt + g φ_z = 0 at z = 0

Bottom:

    φ_z = 0 at z = -h

Wave inlet:
Use the Airy-wave analytical potential as a reference boundary.

## 4. PINN
Network:

    (x,z,t) -> φθ(x,z,t)

Loss:

    L = λf L_PDE + λfs L_FS + λb L_bottom + λi L_inlet

## 5. Validation
Compare PINN and Airy solution using:
- L2 relative error
- RMSE
- maximum absolute error
- wave phase/amplitude error

## 6. Hybrid WEC extension
The next research stage should add a localized WEC boundary and prescribed motion:

    ξ(t) = A_w cos(ωt + θ)

and obtain hydrodynamic pressure from Bernoulli's equation:

    p = -ρ(φ_t + 0.5|∇φ|² + gz) + C(t)

For linear theory, the quadratic velocity term is neglected.

PTO absorbed power for prescribed sinusoidal motion:

    P_PTO = 0.5 c_PTO ω² A_w²

A coupled model can later solve a heave/surge/pitch equation such as:

    m ξ¨ + c_rad ξ˙ + k ξ = F_wave - c_PTO ξ˙

## 7. Limitations
This MVP does not yet model:
- nonlinear free-surface motion
- viscous/turbulent flow
- overtopping
- porous media
- full 3-D geometry
- realistic WEC hydrodynamic coefficients
- two-way coupled structural dynamics

These are deliberate future-work items, not hidden assumptions.

## 8. Four-month milestone
Month 1: math, wave theory, Python/PyTorch, PINN fundamentals.
Month 2: baseline Airy-wave PINN and validation.
Month 3: breakwater/WEC boundary and parameter studies.
Month 4: validation, plots, report, presentation, viva.
