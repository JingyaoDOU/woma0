#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:13:55 2021

@author: sergio
"""
import woma
from importlib import reload
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, jit
from scipy.interpolate import interp1d
import woma.spin_funcs.utils_spin as us

R_earth = 6.371e6  # m
M_earth = 5.9724e24  # kg m^-3

M_f_pE = 0.887
M_f_Th = 0.133
M_pE = M_f_pE * M_earth
M_Th = M_f_Th * M_earth


@jit(nopython=False)
def V_eq_po_from_rho(A1_R, A1_Z, A1_rho, period):

    A1_V_eq = np.zeros(A1_R.shape)
    A1_V_po = np.zeros(A1_Z.shape)

    W = 2 * np.pi / (period * 60 ** 2)

    for i in range(A1_rho.shape[0] - 1):

        if A1_rho[i] == 0:
            break

        delta_rho = A1_rho[i] - A1_rho[i + 1]

        for j in range(A1_V_eq.shape[0]):
            A1_V_eq[j] += us.V_grav_eq(A1_R[j], A1_R[i], A1_Z[i], delta_rho)

        for j in range(A1_V_po.shape[0]):
            A1_V_po[j] += us.V_grav_po(A1_Z[j], A1_R[i], A1_Z[i], delta_rho)

    for i in range(A1_V_eq.shape[0]):
        A1_V_eq[i] += -(1 / 2) * (W * A1_R[i]) ** 2

    return A1_V_eq, A1_V_po


def plot_spinning_profiles(sp):
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    for i, mat in enumerate(sp.A1_mat_id_layer):

        mask = sp.A1_mat_id == mat
        mask_spher = sp.planet.A1_mat_id == mat
        ax[0, 0].scatter(
            sp.A1_R[mask] / R_earth, sp.A1_rho[mask], s=1, label="spinning"
        )
        ax[0, 0].scatter(
            sp.planet.A1_r[mask_spher] / R_earth,
            sp.planet.A1_rho[mask_spher],
            s=5,
            label="spherical",
        )
        ax[0, 0].set_xlabel("R")
        ax[0, 0].set_ylabel("Density")

        ax[0, 1].scatter(sp.A1_R[mask] / R_earth, sp.A1_u[mask], s=5)
        # ax[0,1].scatter(sp.planet.A1_r[mask_spher] / R_earth, sp.planet.A1_u[mask_spher], s=5)
        ax[0, 1].set_xlabel("R")
        ax[0, 1].set_ylabel("specific internal energy")

        ax[1, 0].scatter(sp.A1_R[mask] / R_earth, sp.A1_P[mask], s=5)
        ax[1, 0].scatter(
            sp.planet.A1_r[mask_spher] / R_earth, sp.planet.A1_P[mask_spher], s=5
        )
        ax[1, 0].set_xlabel("R")
        ax[1, 0].set_ylabel("Pressure")

        ax[1, 1].scatter(sp.A1_R[mask] / R_earth, sp.A1_T[mask], s=5)
        # ax[1,1].scatter(sp.planet.A1_r / R_earth, sp.planet.A1_T, s=5)
        ax[1, 1].set_xlabel("R")
        ax[1, 1].set_ylabel("temperature")

    # A1_rho_iron = woma.eos.eos.sesame.A1_rho_ANEOS_Fe85Si15
    # mask = np.logical_and(A1_rho_iron > 7000,
    #                      A1_rho_iron < 9000)
    # ax[0,0].hlines(A1_rho_iron[mask], xmin=0, xmax=0.33, color='red', lw=0.5)

    # ax[0,0].scatter(sp.A1_r_eq / R_earth, sp.A1_rho_eq,
    #                 s=1, label='spinning rho_eq')

    ax[0, 0].legend()
    plt.tight_layout()
    plt.show()


def plot_spherical_profiles(planet):
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    for i, mat in enumerate(planet.A1_mat_id_layer):

        mask_spher = planet.A1_mat_id == mat
        ax[0, 0].scatter(
            planet.A1_r[mask_spher] / R_earth,
            planet.A1_rho[mask_spher],
            s=5,
            label="spherical",
        )
        ax[0, 0].set_xlabel("R")
        ax[0, 0].set_ylabel("Density")

        ax[0, 1].scatter(
            planet.A1_r[mask_spher] / R_earth, planet.A1_u[mask_spher], s=5
        )
        ax[0, 1].set_xlabel("R")
        ax[0, 1].set_ylabel("specific internal energy")

        ax[1, 0].scatter(
            planet.A1_r[mask_spher] / R_earth, planet.A1_P[mask_spher], s=5
        )
        ax[1, 0].set_xlabel("R")
        ax[1, 0].set_ylabel("Pressure")

        ax[1, 1].scatter(planet.A1_r / R_earth, planet.A1_T, s=5)
        ax[1, 1].set_xlabel("R")
        ax[1, 1].set_ylabel("temperature")

    # A1_rho_iron = woma.eos.eos.sesame.A1_rho_ANEOS_Fe85Si15
    # mask = np.logical_and(A1_rho_iron > 7000,
    #                      A1_rho_iron < 9000)
    # ax[0,0].hlines(A1_rho_iron[mask], xmin=0, xmax=0.33, color='red', lw=0.5)

    # ax[0,0].legend()
    plt.tight_layout()
    plt.show()


# %%

planet = woma.Planet(
    A1_mat_layer=["ANEOS_Fe85Si15", "ANEOS_forsterite"],
    A1_T_rho_type=["adiabatic", "adiabatic"],
    # A1_mat_layer    = ["Til_iron", "Til_granite"],
    # A1_T_rho_type   = ["power=0.", "power=0."],
    A1_M_layer=[M_Th * 0.3, M_Th * 0.7],
    P_s=1e5,
    T_s=2000,
)

planet.gen_prof_L2_find_R_R1_given_M1_M2(R_min=0.5 * R_earth, R_max=0.57 * R_earth)
# R_min = 0.5 * R_earth, R_max = 0.62 * R_earth)


# %%
spin_planet = woma.SpinPlanet(
    planet,
    num_prof=1000,
    period=3.2,
    # period=2,
    # check_min_period=True,
    f_iter=0.002,
    num_attempt_1=15,
    num_attempt_2=5,
    verbosity=1,
    tol_density_profile=0.01,
)

# %%
spin_planet_2 = woma.SpinPlanet(
    planet,
    # period=3.2,
    period=4,
    check_min_period=False,
    f_iter=0.002,
    num_attempt_1=15,
    num_attempt_2=5,
    verbosity=1,
    tol_density_profile=0.01,
)

# %%
spin_planet_3 = woma.SpinPlanet(
    planet,
    # period=3.2,
    period=5,
    check_min_period=False,
    f_iter=0.002,
    num_attempt_1=15,
    num_attempt_2=5,
    verbosity=1,
    tol_density_profile=0.01,
)


#%%
plot_spherical_profiles(planet)
plot_spinning_profiles(spin_planet)

#%%
sp = spin_planet

plot_spinning_profiles(sp)

A1_R = sp.A1_R
A1_Z = sp.A1_Z
A1_R = np.hstack((A1_R, A1_R[-1] + A1_R[1]))
A1_Z = np.hstack((A1_Z, A1_Z[-1] + A1_Z[1]))
A1_rho = np.hstack((sp.A1_rho, 0.0))
period = sp.period
A1_P = np.hstack((sp.A1_P, 0.0))

A1_V_eq, A1_V_po = V_eq_po_from_rho(A1_R, A1_Z, A1_rho, period)

F_g = -np.gradient(A1_V_eq, A1_R)
F_h = -np.gradient(A1_P, A1_R) / A1_rho

# F_g = -(A1_V_eq[1:] - A1_V_eq[:-1])/(A1_R[1:] - A1_R[:-1])
# F_h = -(A1_P[1:] - A1_P[:-1])/(A1_R[1:] - A1_R[:-1])/A1_rho[1:]

plt.figure()
plt.scatter(A1_R[:] / R_earth, F_g, label="F_g + F_c", s=1)
plt.scatter(A1_R[:] / R_earth, F_h, label="F_h", s=1)
plt.scatter(A1_R[:] / R_earth, F_h + F_g, label="F_g + F_c + F_h", s=1)
plt.xlabel("R eq")
plt.ylabel("F")
plt.legend()
plt.show()


plt.figure()
plt.scatter(A1_R[:] / R_earth, A1_V_eq, label="V", s=1)
plt.xlabel("R po")
plt.ylabel("V")
plt.legend()
plt.show()

#%%
sp = spin_planet

plot_spinning_profiles(sp)

A1_R = sp.A1_R
A1_Z = sp.A1_Z
A1_mat_id = sp.A1_mat_id
A1_R = np.hstack((A1_R, A1_R[-1] + A1_R[1]))
A1_Z = np.hstack((A1_Z, A1_Z[-1] + A1_Z[1]))
A1_mat_id = np.hstack((A1_mat_id, 0.0))
A1_rho = np.hstack((sp.A1_rho, 0.0))
period = sp.period
A1_P = np.hstack((sp.A1_P, 0.0))

A1_V_eq, A1_V_po = V_eq_po_from_rho(A1_R, A1_Z, A1_rho, period)

F_g = -np.gradient(A1_V_eq, A1_R)

mask_1 = A1_mat_id == sp.A1_mat_id_layer[0]
F_h_1 = -np.gradient(A1_P[mask_1], A1_R[mask_1]) / A1_rho[mask_1]
F_h = (
    -(A1_P[mask_1][1:] - A1_P[mask_1][:-1])
    / (A1_R[mask_1][1:] - A1_R[mask_1][:-1])
    / A1_rho[mask_1][1:]
)
mask_2 = A1_mat_id == sp.A1_mat_id_layer[1]
F_h_2 = -np.gradient(A1_P[mask_2], A1_R[mask_2]) / A1_rho[mask_2]


# F_g = -(A1_V_eq[1:] - A1_V_eq[:-1])/(A1_R[1:] - A1_R[:-1])
# F_h = -(A1_P[1:] - A1_P[:-1])/(A1_R[1:] - A1_R[:-1])/A1_rho[1:]

plt.figure()
plt.scatter(A1_R[:] / R_earth, F_g, label="F_g + F_c", s=1)
plt.scatter(A1_R[mask_1] / R_earth, F_h_1, label="F_h_1", s=1)
plt.scatter(A1_R[mask_2] / R_earth, F_h_2, label="F_h_2", s=1)
# plt.scatter(A1_R[:]/R_earth, F_h_1 + F_h_2 + F_g, label="F_g + F_c + F_h", s=1)
plt.xlabel("R eq")
plt.ylabel("F")
plt.legend()
plt.show()


plt.figure()
plt.scatter(A1_R[:] / R_earth, A1_V_eq, label="V", s=1)
plt.xlabel("R po")
plt.ylabel("V")
plt.legend()
plt.show()

#%%%
P_i = spin_planet.planet.P_0
mat_id = spin_planet.A1_mat_id_layer[0]
T_rho_type_id = spin_planet.A1_T_rho_type_id[0]
T_rho_args = spin_planet.planet.A1_T_rho_args[0]
rho_s = 3054.5
rho_0 = 9000

rho = woma.eos.eos.find_rho(P_i, mat_id, T_rho_type_id, T_rho_args, rho_s * 0.1, rho_0,)

rho = spin_planet.planet.rho_0

T = woma.eos.T_rho.T_rho(rho, T_rho_type_id, T_rho_args, mat_id,)
u = woma.eos.eos.u_rho_T(rho, T, mat_id)
P_f = woma.eos.eos.P_u_rho(u, rho, mat_id)

print((P_f - P_i) / P_f)

#%%
import numpy as np
import matplotlib.pyplot as plt

A1_i = np.linspace(0, 5, 1000)


def f1(A1_i):
    A1_out = np.copy(A1_i)
    A1_out[A1_i < 0.1] = 0.1
    A1_out = 1 / (A1_out ** 2)
    return A1_out


def f2(A1_i):
    A1_out = np.copy(A1_i)
    # A1_out = 1/(1 + A1_out)**np.log2(10)
    A1_out = 1 / (1 + A1_out) ** 2
    return A1_out


def f3(A1_i):
    A1_out = np.copy(A1_i)
    A1_out = np.exp(-(A1_out ** 2))
    # A1_out = 1/(1 + A1_out)**2
    return A1_out


A1_i_out1 = f1(A1_i)
A1_i_out2 = f2(A1_i)
A1_i_out3 = f3(A1_i)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(A1_i, A1_i_out1)
# ax.plot(A1_i, A1_i_out2)
ax.plot(A1_i, A1_i_out3)
ax.set_yscale("log")
ax.set_xlabel("I")
ax.set_ylabel("f(I)")
plt.show()


#%%

woma.load_eos_tables()

T = 1000
mat_id = 400
A1_rho = np.linspace(0, 10000, 100)
A1_P = np.zeros_like(A1_rho)

for i, rho in enumerate(A1_rho):
    A1_P[i] = woma.eos.eos.P_T_rho(T, rho, mat_id)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(A1_P, A1_rho)
# ax.plot(A1_i, A1_i_out2)
ax.set_xscale("log")
ax.set_xlim(A1_P[1], A1_P[-1])
ax.set_xlabel("P")
ax.set_ylabel("rho")
plt.show()
