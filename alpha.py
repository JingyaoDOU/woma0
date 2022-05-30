#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:42:05 2021

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


def compute_Nngb(kernel, eta=1.2348):
    # get gamma=H/h
    if kernel == "s3":  # cubic spline
        gamma = 1.825742
    elif kernel == "wc6":  # wendland C6
        gamma = 2.449490
    else:
        raise ValueError("Kernel not available")

    N_ngb = 4 * np.pi / 3 * (gamma * eta) ** 3
    return N_ngb


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


@njit
def create_half_grid_sphere(N_1D=41, r=2):
    A1_x = np.linspace(-r, r, N_1D)
    A1_y = np.linspace(-r, r, N_1D)
    A1_z = np.linspace(-r, r, N_1D)

    A2_pos = []
    A1_r = []
    for i, x in enumerate(A1_x):
        for j, y in enumerate(A1_y):
            for k, z in enumerate(A1_z):
                radius = np.sqrt(x * x + y * y + z * z)
                if radius <= r and z <= 0:
                    A2_pos.append([x, y, z])
                    A1_r.append(radius)

    A2_pos = np.array(A2_pos)
    A1_r = np.array(A1_r)

    return A2_pos, A1_r


#%% wc6

A1_N1D = np.arange(3, 15, 2)
A1_N_ngb = []
A1_I = []

compute_A1_W = compute_A1_W_wendland_C6
# H = 1.825742 # cubic spline
H = 2.449490  # wendland c6

for N1D in A1_N1D:
    # generate half sphere grid of particles
    A2_pos, A1_r = create_half_grid_sphere(N_1D=N1D, r=H)

    # assert 0,0,0 is in A2_pos
    assert np.sum(np.sum(A2_pos == 0, axis=1) == 3)

    # compute total number of particles and save it
    N = A1_r.shape[0]
    A1_N_ngb.append(N)

    # compute I and save it
    I = compute_I(A2_pos, A1_r, compute_A1_kernel=compute_A1_W)
    A1_I.append(I)

# convert to numpy arrays
A1_N_ngb = np.array(A1_N_ngb)
A1_I = np.array(A1_I)

# plot results
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(A1_N_ngb, A1_I, s=20)
ax.plot(A1_N_ngb, A1_I)

# plot vertical line
plt.axvline(x=compute_Nngb("wc6"))

# ax.set_yscale('log')
ax.set_ylabel("I")
ax.set_xlabel("N neig")

plt.show()


#%% s3

A1_N1D = np.arange(3, 15, 2)
A1_N_ngb = []
A1_I = []

compute_A1_W = compute_A1_W_cubic_spline
H = 1.825742  # cubic spline
# H = 2.449490 # wendland c6

for N1D in A1_N1D:
    # generate half sphere grid of particles
    A2_pos, A1_r = create_half_grid_sphere(N_1D=N1D, r=H)

    # assert 0,0,0 is in A2_pos
    assert np.sum(np.sum(A2_pos == 0, axis=1) == 3)

    # compute total number of particles and save it
    N = A1_r.shape[0]
    A1_N_ngb.append(N)

    # compute I and save it
    I = compute_I(A2_pos, A1_r, compute_A1_kernel=compute_A1_W)
    A1_I.append(I)

# convert to numpy arrays
A1_N_ngb = np.array(A1_N_ngb)
A1_I = np.array(A1_I)

# plot results
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(A1_N_ngb, A1_I, s=20)
ax.plot(A1_N_ngb, A1_I)

# plot vertical line
plt.axvline(x=compute_Nngb("s3"))

# ax.set_yscale('log')
ax.set_ylabel("I")
ax.set_xlabel("N neig")

plt.show()
