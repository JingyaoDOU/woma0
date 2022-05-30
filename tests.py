#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:30:13 2020

@author: sergio
"""
# %%
import numpy as np
import woma
import seagen
import matplotlib.pyplot as plt

R_earth = 6.371e6  # m
M_earth = 5.972e24  # kg m^-3

# %%

planet = woma.Planet(
    A1_mat_layer=["Til_iron", "Til_basalt"],
    A1_T_rho_type=["power=2", "power=2"],
    P_s=1e5,
    T_s=1000,
    M=M_earth,
    R=R_earth,
)

# Generate the profiles
planet.gen_prof_L2_find_R1_given_M_R(verbosity=1)

# %%
spin_planet = woma.SpinPlanet(
    planet=planet,
    period=6,
    verbosity=1,
)  # h

# %%
N = 1e5
particles = woma.ParticlePlanet(spin_planet, N, N_ngb=48)
particles_seagen = woma.ParticlePlanet(planet, N, N_ngb=48)

particles


# %%
A1_rho_unique = np.unique(particles.A1_rho)
A1_N = np.zeros_like(A1_rho_unique)
for i, rho in enumerate(A1_rho_unique):
    A1_N[i] = np.sum(particles.A1_rho == rho)


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(particles.A1_rho, particles.A1_m / M_earth, s=4)
ax.set_xlabel("density [kg m ^-3]")
ax.set_ylabel("particle mass [M_earth]")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(A1_rho_unique, A1_N, s=4)
ax.set_xlabel("density [kg m ^-3]")
ax.set_ylabel("number of particles")
plt.show()
