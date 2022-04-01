#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:40:06 2021

@author: sergio
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# default h = 1


@njit
def compute_A1_W_cubic_spline(A1_r, h=1):
    C = 16 / np.pi
    gamma = 1.825742
    H = gamma * h

    A1_u = A1_r / H

    A1_W = np.zeros_like(A1_u)

    mask = np.logical_and(A1_u < 1 / 2, A1_u >= 0)
    A1_W[mask] = 3 * A1_u[mask] ** 3 - 3 * A1_u[mask] ** 2 + 1 / 2

    mask = np.logical_and(A1_u >= 1 / 2, A1_u < 1)
    A1_W[mask] = -A1_u[mask] ** 3 + 3 * A1_u[mask] ** 2 - 3 * A1_u[mask] + 1

    return C * A1_W / H ** 3


@njit
def compute_A1_W_wendland_C2(A1_r, h=1):
    C = 21 / 2 / np.pi
    gamma = 1.936492
    H = gamma * h

    A1_u = A1_r / H

    A1_W = np.zeros_like(A1_u)

    mask = np.logical_and(A1_u >= 0, A1_u <= 1)

    A1_W[mask] += 4 * A1_u[mask] ** 5
    A1_W[mask] += -15 * A1_u[mask] ** 4
    A1_W[mask] += 20 * A1_u[mask] ** 3
    A1_W[mask] += -10 * A1_u[mask] ** 2
    A1_W[mask] += 1

    return C * A1_W / H ** 3


@njit
def compute_A1_W_wendland_C4(A1_r, h=1):
    C = 495 / 32 / np.pi
    gamma = 2.207940
    H = gamma * h

    A1_u = A1_r / H

    A1_W = np.zeros_like(A1_u)

    mask = np.logical_and(A1_u >= 0, A1_u <= 1)

    A1_W[mask] += 35 / 3 * A1_u[mask] ** 8
    A1_W[mask] += -64 * A1_u[mask] ** 7
    A1_W[mask] += 140 * A1_u[mask] ** 6
    A1_W[mask] += -448 / 3 * A1_u[mask] ** 5
    A1_W[mask] += 70 * A1_u[mask] ** 4
    A1_W[mask] += -28 / 3 * A1_u[mask] ** 2
    A1_W[mask] += 1

    return C * A1_W / H ** 3


@njit
def compute_A1_W_wendland_C6(A1_r, h=1):
    C = 1365 / 64 / np.pi
    gamma = 2.449490
    H = gamma * h

    A1_u = A1_r / H

    A1_W = np.zeros_like(A1_u)

    mask = np.logical_and(A1_u >= 0, A1_u <= 1)

    A1_W[mask] += 32 * A1_u[mask] ** 11
    A1_W[mask] += -231 * A1_u[mask] ** 10
    A1_W[mask] += 704 * A1_u[mask] ** 9
    A1_W[mask] += -1155 * A1_u[mask] ** 8
    A1_W[mask] += 1056 * A1_u[mask] ** 7
    A1_W[mask] += -462 * A1_u[mask] ** 6
    A1_W[mask] += 66 * A1_u[mask] ** 4
    A1_W[mask] += -11 * A1_u[mask] ** 2
    A1_W[mask] += 1

    return C * A1_W / H ** 3


def compute_I(A2_pos, A1_r, h=1, compute_A1_kernel=compute_A1_W_cubic_spline):

    # N = len(A2_pos)
    A1_W = np.sqrt(compute_A1_kernel(A1_r, h))
    I = np.sum(A2_pos * A1_W.reshape(-1, 1), axis=0)
    I = np.sqrt(np.sum(I ** 2))
    I = I / h
    I = I / np.sum(A1_W)

    # I = np.sum(A2_pos, axis=0)
    # I = np.sqrt(np.sum(I**2))
    # I = I/(N*h)

    return I


@njit
def create_grid_sphere(N_1D=41, r=2):
    A1_x = np.linspace(-r, r, N_1D)
    A1_y = np.linspace(-r, r, N_1D)
    A1_z = np.linspace(-r, r, N_1D)

    A2_pos = []
    A1_r = []
    for i, x in enumerate(A1_x):
        for j, y in enumerate(A1_y):
            for k, z in enumerate(A1_z):
                radius = np.sqrt(x * x + y * y + z * z)
                if radius <= r:
                    A2_pos.append([x, y, z])
                    A1_r.append(radius)

    A2_pos = np.array(A2_pos)
    A1_r = np.array(A1_r)

    return A2_pos, A1_r


def create_data(N_1D, r=2, h=1):

    A2_pos, A1_r = create_grid_sphere(N_1D, r)
    N = len(A1_r)
    A1_z = np.linspace(r, -r, N_1D)

    A1_I = []
    A1_f_volume = []

    for k, z in enumerate(A1_z[:-2]):
        mask = A2_pos[:, 2] <= z
        A2_pos_masked = A2_pos[mask]
        A1_r_masked = A1_r[mask]

        vol = len(A2_pos_masked) / N
        if len(A2_pos_masked) <= 2:
            break
        else:
            I = compute_I(A2_pos_masked, A1_r_masked, h)

        A1_f_volume.append(vol)
        A1_I.append(I)

    A1_I = np.array(A1_I)
    A1_f_volume = np.array(A1_f_volume)

    return A1_f_volume, A1_I


#%%
# A1_N1D = np.arange(3, 21, step=2)
N1D = 11
A1_N_neig = []
A1_I = []

compute_A1_W = compute_A1_W_wendland_C6
H = 1.825742  # cubic spline
H = 2.207940  # wendland c4
H = 2.449490  # wendland c4

for r in np.linspace(H, 5 * H, 1000):
    A2_pos, A1_r = create_grid_sphere(N_1D=N1D, r=r)
    mask = np.logical_and(A2_pos[:, 2] <= 0.0, compute_A1_W(A1_r) > 0.0)
    A2_pos = A2_pos[mask]
    A1_r = A1_r[mask]

    assert np.sum(np.sum(A2_pos == np.array([0.0, 0.0, 0.0]), axis=1) == 3) == 1
    A1_N_neig.append(len(A2_pos))
    A1_I.append(compute_I(A2_pos, A1_r))

A1_N_neig = np.array(A1_N_neig)
A1_I = np.array(A1_I)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(A1_N_neig, A1_I, s=10)

# ax.set_yscale('log')
ax.set_ylabel("I")
ax.set_xlabel("N neig")

plt.show()

#%%
A1_f_volume, A1_I = create_data(121, r=2, h=1)
#%%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(A1_I, A1_f_volume, s=10)

# ax.set_yscale('log')
ax.set_ylabel("fractional volume occupied")
ax.set_xlabel("I")

plt.show()

#%% visualize W
A1_r = np.linspace(0, 2.5, 1000)
A1_W_cubic = compute_A1_W_cubic_spline(A1_r)
A1_W_cubic_sqrt = np.sqrt(compute_A1_W_cubic_spline(A1_r))
A1_W_C2 = compute_A1_W_wendland_C2(A1_r)
A1_W_C4 = compute_A1_W_wendland_C4(A1_r)
A1_W_C6 = compute_A1_W_wendland_C6(A1_r)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(A1_r, A1_W_cubic / A1_W_cubic[0], s=10)
ax.scatter(A1_r, A1_W_cubic_sqrt / A1_W_cubic_sqrt[0], s=10)
# ax.scatter(A1_r, A1_W_C2/A1_W_C2[0], s=10)
# ax.scatter(A1_r, A1_W_C4/A1_W_C4[0], s=10)
# ax.scatter(A1_r, A1_W_C6/A1_W_C6[0], s=10)

# ax.set_yscale('log')
# ax.set_ylim(1e-10, 2e0)
ax.set_ylabel("W")
ax.set_xlabel("r")

plt.show()
