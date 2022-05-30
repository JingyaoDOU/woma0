#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:00:22 2021

@author: sergio
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

import seagen
import woma


# all quantities in this file are in SI
L_em = 3.5e34
R_earth = 6_371_000
R_moon = 1_737_400
M_earth = 5.9724e24
G = 6.67408e-11  # m^3 kg^-1 s^-2


class Simulation:
    def __init__(
        self,
        dt=0.1,  # timestep
        t_init=0.0,  # initial time
        t_end=10000,  # end time
        # all particles with m lower than this value
        # will not exhert any gravity force to the rest of particles
        min_m_gravity=0.00001 * M_earth,
        # if two particles are closer than this value
        # they will not exhert any gravity force to each other
        grav_soft_length=2 * R_moon,
        dt_save=100,  # save data every dt_save seconds
    ):
        self.dt = dt
        self.t_init = t_init
        self.t_end = t_end
        self.min_m_gravity = min_m_gravity
        self.grav_soft_length = grav_soft_length
        self.dt_save = dt_save

        self.save = []
        self.N_steps = int((self.t_end - self.t_init) / self.dt)
        self.N_steps_save = int(self.dt_save / self.dt)


class Particle:
    def __init__(
        self, A1_pos=None, A1_vel=None, A1_acc=None, A1_F=None, m=None,
    ):

        self.A1_pos = np.array(A1_pos, dtype="float")
        self.A1_vel = np.array(A1_vel, dtype="float")
        self.A1_acc = np.zeros(3)
        self.A1_F = np.zeros(3)
        self.m = m


def update_forces(A1_picles, simulation):

    # set all forces to 0
    for i, picle_i in enumerate(A1_picles):
        picle_i.A1_F = np.zeros(3)

    # loop over all particles twice
    for i, picle_i in enumerate(A1_picles):
        for j, picle_j in enumerate(A1_picles):

            # F_ij = -F_ji
            if j >= i:
                continue

            # ignore F_ij if m_i and m_j < min_m_gravity
            if (
                picle_i.m < simulation.min_m_gravity
                and picle_j.m < simulation.min_m_gravity
            ):
                # print("min_m_grav")
                continue

            # compute distance between particle i and particle j
            A1_r_ij = picle_i.A1_pos - picle_j.A1_pos
            r_ij = np.linalg.norm(A1_r_ij)

            # ignore if r_ij < grav_soft_length
            if r_ij < simulation.grav_soft_length:
                # print("grav_soft_len")
                continue

            # compute force
            A1_F = -G * picle_i.m * picle_j.m * A1_r_ij / (r_ij ** 3)
            picle_i.A1_F += A1_F
            picle_j.A1_F += -A1_F


def update_velocities_and_positions(A1_picles, simulation):
    dt = simulation.dt
    for i, picle_i in enumerate(A1_picles):
        picle_i.A1_vel = picle_i.A1_vel + picle_i.A1_F * dt / picle_i.m
        picle_i.A1_pos = picle_i.A1_pos + picle_i.A1_vel * dt


def run_simulation(A1_picles, simulation):
    for i in range(simulation.N_steps):

        # save data
        if i % simulation.N_steps_save == 0:
            simulation.save.append(copy.deepcopy(A1_picles))

        # update forces
        update_forces(A1_picles, simulation)

        # update positions
        update_velocities_and_positions(A1_picles, simulation)


# def plot_trajectories(simulation, ax):
#    for i, A1_picles in enumerate(simulation.save):
#        for j, picle in enumerate(A1_picles):
#            ax.scatter(picle.A1_pos[0]/R_earth, picle.A1_pos[1]/R_earth, c='black')


def plot_trajectories(simulation, ax):
    N_picles = len(simulation.save[0])
    for j in range(N_picles):
        A2_pos = np.zeros((len(simulation.save), 2))
        for i, A1_picles in enumerate(simulation.save):
            A2_pos[i] = A1_picles[j].A1_pos[:2]
        ax.scatter(A2_pos[:, 0] / R_earth, A2_pos[:, 1] / R_earth, s=1)


#%% toy example

# define particles
picle_1 = Particle(A1_pos=[0, 0, 0], A1_vel=[0, 0, 0], m=1 * M_earth)

picle_2 = Particle(A1_pos=[0, -5 * R_earth, 0], A1_vel=[2800, 0, 0], m=0.002 * M_earth)

picle_3 = Particle(
    A1_pos=[0, -5 * R_earth + R_moon / 2, 0], A1_vel=[2800, 0, 0], m=0.002 * M_earth
)

picle_4 = Particle(
    A1_pos=[0, -5 * R_earth - R_moon / 2, 0], A1_vel=[2800, 0, 0], m=0.002 * M_earth
)

picle_5 = Particle(
    A1_pos=[0, -5 * R_earth + R_moon / 4, 0], A1_vel=[2800, 0, 0], m=0.002 * M_earth
)

picle_6 = Particle(
    A1_pos=[0, -5 * R_earth - R_moon / 4, 0], A1_vel=[2800, 0, 0], m=0.002 * M_earth
)

A1_picles = [picle_1, picle_2, picle_3, picle_4, picle_5, picle_6]

# define simulation parameters
simulation = Simulation(dt=1, t_end=25_000, dt_save=100)

# run simulation
run_simulation(A1_picles, simulation)

#%% plot results

# define figure
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# plot circle at R_earth and R_roche=3*R_earth
circle_size = 1
circle = plt.Circle((0, 0), circle_size, color="black", fill=False)
ax.add_artist(circle)

circle_size = 3
circle = plt.Circle((0, 0), circle_size, color="grey", fill=False)
ax.add_artist(circle)

# plot trajectories
plot_trajectories(simulation, ax)

# limits
lim = 7
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")

# show plot
plt.show()
