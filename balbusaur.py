################################################################################
#
#   File: balbusaur.py
# Author: Ben Ryan <brryan@lanl.gov>
#   Date: 2019 Dec 1
#  Brief: Symbolic linearization and symbolic/numeric evaluation of eigenmodes
#         for an arbitrary set of coupled hyperbolic partial differential
#         equations.
#         Original sagemath method due to Mani Chandra.
#   Note: Copyright (C) 2019-2021 Triad National Security, LLC.
#         All rights reserved.
#
################################################################################

from sympy import *
import mpmath as math
import sys
import numpy as np
init_printing()

# Adapted from https://stackoverflow.com/questions/22857162/multivariate-taylor-approximation-in-sympy
def multivar_taylor(func, x, x0):
    nvar = len(x)
    hlist = ['__h' + str(n+1) for n in range(nvar)]
    command = "symbols('" + '  '.join(hlist) + "')"
    hvar = eval(command)
    t = symbols('t')
    loc_func = func
    for i in range(nvar):
      locvar = x[i]
      locsub = x0[i] + t*hvar[i]
      loc_func = loc_func.subs(locvar, locsub)

    g = 0
    g += loc_func.diff(t,i).subs(t,0)*t

    for i in range(nvar):
      g = g.subs(hlist[i], x[i] - x0[i])

    g = g.subs(t, 1)
    return g

################################################################################
# USER INPUT
################################################################################
# Set background values
rho0_num = 1.
ug0_num = 1.e-2

# Set adiabatic index
gamma_num = 5./3.

# Set dimensionless parameters
beta_m_num = 1. # Gas/magnetic pressure
beta_r_num = 1. # Gas/radiation pressure
opticaldepth_num = 20
################################################################################
# END USER INPUT
################################################################################

# Numerical values for constants
P0_num      = (gamma_num - 1.)*ug0_num                   # Mean pressure
T0_num      = P0_num/rho0_num                            # Mean temperature
kbol_num    = 1.                                         # Boltzmann constant
c_num       = 1.                                         # Speed of light
a_R_num     = 3.*rho0_num**4/(beta_r_num*P0_num**3)
ur0_num     = a_R_num*T0_num**4.
k1_num      = 2.*pi                                      # x1 wavenumber
k2_num      = 0                                          # x2 wavenumber
kappa_a_num = opticaldepth_num*k1_num/(2.*pi*rho0_num)/2   # Absorption opacity
kappa_s_num = opticaldepth_num*k1_num/(2.*pi*rho0_num)/2   # Scattering opacity
pi_num      = 3.14159265359
alpha_num   = (gamma_num - 1.)
hpl_num     = (8.*pi_num**5.*kbol_num**4./(15.*c_num**3.*a_R_num))**(1./3.)
ke0_num     = P0_num/rho0_num**gamma_num
B0_num      = np.sqrt(2.*P0_num/beta_m_num)
B10_num     = np.sqrt(2.)/2.*B0_num
B20_num     = np.sqrt(2.)/2.*B0_num

# Report problem parameters
print('\nParameters:')
print('beta_r:  %e' % beta_r_num)
print('beta_m:  %e' % beta_m_num)
print('aR:      %e' % a_R_num)
print('kappa_a: %e' % kappa_a_num)
print('kappa_s: %e' % kappa_s_num)
print('')

t, omega, k1, k2, k3 = var('t, omega, k1, k2, k3')
gamma, a_R, kappa_a, kappa_s, alpha = var('gamma, a_R, kappa_a, kappa_s, alpha')
rho0, ug0, u10, u20, u30, B10, B20, B30 = var('rho0, ug0, u10, u20, u30, B10, B20, B30')
ur0, F10, F20, F30 = var('ur0, F10, F20, F30')
delta_rho, delta_ug, delta_u1, delta_u2, delta_u3, delta_B1, delta_B2, delta_B3 = \
    var('delta_rho, delta_ug, delta_u1, delta_u2, delta_u3, delta_B1, delta_B2, delta_B3')
delta_ur, delta_F1, delta_F2, delta_F3 = \
    var('delta_ur, delta_F1, delta_F2, delta_F3')
delta_rho_dt, delta_ug_dt, delta_u1_dt, delta_u2_dt, delta_u3_dt = \
    var('delta_rho_dt, delta_ug_dt, delta_u1_dt, delta_u2_dt, delta_u3_dt')
delta_B1_dt, delta_B2_dt, delta_B3_dt = \
    var('delta_B1_dt, delta_B2_dt, delta_B3_dt')
delta_ur_dr, delta_F1_dt, delta_F2_dt, delta_F3_dt = \
    var('delta_ur_dt, delta_F1_dt, delta_F2_dt, delta_F3_dt')

rho = rho0 + delta_rho
ug  = ug0 + delta_ug
u1  = delta_u1
u2  = delta_u2
u3  = 0
B1  = B10
B2  = B20 + delta_B2
B3  = 0
ur  = ur0 + delta_ur
F1  = delta_F1
F2  = delta_F2
F3  = 0

gcon = Matrix(  [[-1, 0, 0, 0],
                 [ 0, 1, 0, 0],
                 [ 0, 0, 1, 0],
                 [ 0, 0, 0, 1]])
gcov = gcon.inv()
gamma = sqrt(1 +    gcov[1,1]*u1*u1 + gcov[2,2]*u2*u2 + gcov[3,3]*u3*u3
               + 2*(gcov[1,2]*u1*u2 + gcov[1,3]*u1*u3 + gcov[2,3]*u2*u3))
ucon = [ gamma, u1, u2, u3]
ucov = [-gamma, u1, u2, u3]
bcon0 = B1*ucov[1] + B2*ucov[2] + B3*ucov[3]
bcon = [bcon0, (B1 + bcon0*ucon[1])/ucon[0], (B2 + bcon0*ucon[2])/ucon[0], (B3 + bcon0*ucon[3])/ucon[0]]
bcov = [-bcon[0], bcon[1], bcon[2], bcon[3]]
bsq = bcon[0]*bcov[0] + bcon[1]*bcov[1] + bcon[2]*bcov[2] + bcon[3]*bcov[3]
Fcon = [-(F1*ucov[1] + F2*ucov[2])/ucov[0], F1, F2, 0]
Fcov = [-Fcon[0], F1, F2, 0]

P = alpha*ug # Suppresses expansion of alpha to the fourth power
T = P/rho

def delta(mu, nu):
    if (mu == nu):
      return 1
    else:
      return 0

def Tud(mu, nu):
    return (rho + ug + P + bsq)*ucon[mu]*ucov[nu] + (P + bsq/2)*delta(mu,nu) - bcon[mu]*bcov[nu]
def Rud(mu, nu):
    return 4/3*ur*ucon[mu]*ucov[nu] + 1/3*ur*delta(mu,nu) + Fcon[mu]*ucov[nu] + ucon[mu]*Fcov[nu]
def Gcov(nu):
    return rho*kappa_a*(ur - a_R*T**4)*ucov[nu] + rho*(kappa_a + kappa_s)*Fcov[nu]
def Induction(i, j):
    return bcon[i]*ucon[j] - bcon[j]*ucon[i]

def linearize(term):
  if term == 0:
    return 0
  x = [delta_rho, delta_ug, delta_u1, delta_u2, delta_u3, delta_B1, delta_B2, delta_B3, delta_ur, delta_F1, delta_F2, delta_F3]
  x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  return multivar_taylor(term, x, x0)

def get_coeff(term, var):
  coeffs = Poly(term, var).all_coeffs()
  if len(coeffs) == 2:
    return coeffs[0]
  else:
    return 0

def d_dt(term):
  term = linearize(term)
  expr = 0
  expr += get_coeff(term, delta_rho)*delta_rho_dt
  expr += get_coeff(term, delta_ug)*delta_ug_dt
  expr += get_coeff(term, delta_u1)*delta_u1_dt
  expr += get_coeff(term, delta_u2)*delta_u2_dt
  expr += get_coeff(term, delta_u3)*delta_u3_dt
  expr += get_coeff(term, delta_B1)*delta_B1_dt
  expr += get_coeff(term, delta_B2)*delta_B2_dt
  expr += get_coeff(term, delta_B3)*delta_B3_dt
  expr += get_coeff(term, delta_ur)*delta_ur_dt
  expr += get_coeff(term, delta_F1)*delta_F1_dt
  expr += get_coeff(term, delta_F2)*delta_F2_dt
  expr += get_coeff(term, delta_F3)*delta_F3_dt
  return expr

def d_dX1(term):
  term = linearize(term)
  expr = 0
  expr += get_coeff(term, delta_rho)*I*k1*delta_rho
  expr += get_coeff(term, delta_ug)*I*k1*delta_ug
  expr += get_coeff(term, delta_u1)*I*k1*delta_u1
  expr += get_coeff(term, delta_u2)*I*k1*delta_u2
  expr += get_coeff(term, delta_u3)*I*k1*delta_u3
  expr += get_coeff(term, delta_B1)*I*k1*delta_B1
  expr += get_coeff(term, delta_B2)*I*k1*delta_B2
  expr += get_coeff(term, delta_B3)*I*k1*delta_B3
  expr += get_coeff(term, delta_ur)*I*k1*delta_ur
  expr += get_coeff(term, delta_F1)*I*k1*delta_F1
  expr += get_coeff(term, delta_F2)*I*k1*delta_F2
  expr += get_coeff(term, delta_F3)*I*k1*delta_F3
  return expr

def d_dX2(term):
  term = linearize(term)
  expr = 0
  return expr

def d_dX3(term):
  term = linearize(term)
  expr = 0
  return expr

Eqn_rho = linearize(d_dt(rho*ucon[0]) + d_dX1(rho*ucon[1]))
Eqn_ug  = linearize(d_dt(Tud(0,0)) + d_dX1(Tud(1,0)) + d_dX2(Tud(2,0)) + d_dX3(Tud(3,0)) - Gcov(0))
Eqn_u1  = linearize(d_dt(Tud(0,1)) + d_dX1(Tud(1,1)) + d_dX2(Tud(2,1)) + d_dX3(Tud(3,1)) - Gcov(1))
Eqn_u2  = linearize(d_dt(Tud(0,2)) + d_dX1(Tud(1,2)) + d_dX2(Tud(2,2)) + d_dX3(Tud(3,2)) - Gcov(2))
Eqn_u3  = linearize(d_dt(Tud(0,3)) + d_dX1(Tud(1,3)) + d_dX2(Tud(2,3)) + d_dX3(Tud(3,3)) - Gcov(3))
Eqn_B1  = linearize(d_dt(B1) + d_dX2(Induction(1,2)) + d_dX3(Induction(1,3)))
Eqn_B2  = linearize(d_dt(B2) + d_dX1(Induction(2,1)) + d_dX3(Induction(2,3)))
Eqn_B3  = linearize(d_dt(B3) + d_dX1(Induction(3,1)) + d_dX2(Induction(3,2)))
Eqn_ur  = linearize(d_dt(Rud(0,0)) + d_dX1(Rud(1,0)) + d_dX2(Rud(2,0)) + d_dX3(Rud(3,0)) + Gcov(0))
Eqn_F1  = linearize(d_dt(Rud(0,1)) + d_dX1(Rud(1,1)) + d_dX2(Rud(2,1)) + d_dX3(Rud(3,1)) + Gcov(1))
Eqn_F2  = linearize(d_dt(Rud(0,2)) + d_dX1(Rud(1,2)) + d_dX2(Rud(2,2)) + d_dX3(Rud(3,2)) + Gcov(2))
Eqn_F3  = linearize(d_dt(Rud(0,3)) + d_dX1(Rud(1,3)) + d_dX2(Rud(2,3)) + d_dX3(Rud(3,3)) + Gcov(3))

system = [Eqn_rho, Eqn_ug, Eqn_u1, Eqn_u2, Eqn_B2, Eqn_ur, Eqn_F1, Eqn_F2]
delta_vars = [delta_rho, delta_ug, delta_u1, delta_u2, delta_B2, delta_ur, delta_F1, delta_F2]
delta_vars_dt = [delta_rho_dt, delta_ug_dt, delta_u1_dt, delta_u2_dt, delta_B2_dt, delta_ur_dt,
                 delta_F1_dt, delta_F2_dt]
var_names = ['drho', 'dug', 'du1', 'du2', 'dB2', 'dur', 'dF1', 'dF2']

#ans = linsolve(simplify(system), delta_vars_dt) # simplify() call fails
print(system)
print(delta_vars_dt)
ans = linsolve(system, delta_vars_dt)
solns_delta_vars_dt = ans.args[0]
def fmat(i,j):
  return solns_delta_vars_dt[j]
X = Matrix(1,len(solns_delta_vars_dt), fmat)
def fmat(i,j):
  return delta_vars[j]
Y = Matrix(1, len(delta_vars), fmat)
M = X.jacobian(Y)

beta = 0.5
gam = 1/sqrt(1 - beta**2)
M_num = M.subs(rho0, rho0_num).subs(alpha, alpha_num).subs(k1, k1_num).subs(ug0, ug0_num).subs(pi, pi_num).subs(u10, gam*beta).subs(kappa_a, kappa_a_num)
M_num = M_num.subs(ur0, ur0_num).subs(kappa_s, kappa_s_num).subs(a_R, a_R_num)
M_num = M_num.subs(B10, B10_num).subs(B20, B20_num)
M_num = M_num.evalf()

f = lambdify((), M_num, modules='numpy')
w, v = np.linalg.eig(f())

print('\nModes:')
for val, vecs in zip(w, v.T):
  print('Eigenvalue:')
  print(val)
  print('Eigenvector:')
  for n, vec in enumerate(vecs):
    print(var_names[n] + ': ', vec)
  print('')
