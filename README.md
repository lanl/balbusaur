Balbusaur: Eigenmode Solver for Linearized, Hyperbolic PDEs

Balbusaur is a script for linearizing a set of coupled hyperbolic partial
differential equations and solving for the eigenmodes of the resulting linear
system. The input is the set of governing equations and a functional form of the
perturbation, and the output, the eigenmodes of the linearized system, can be
either analytic or numerical. These eigenmodes are valuable for testing
numerical methods for solving the full system of equations, as they are an exact
solution at linear order.

The script is written in python and uses the sympy library to perform the
symbolic manipulations. The allowable space of input equations is general to
problems in continuous media, but the particular realization of the code
includes as a demonstration the equations of relativistic radiation
magnetohydrodynamics, with the radiation closed with the Eddington tensor
approximation.

Questions/comments: brryan@lanl.gov

The original method is due to Mani Chandra.
