"""
WoMa 3 layer spherical functions 

Write by Jingyao March, 2022
"""

import numpy as np
from numba import njit
import warnings

warnings.filterwarnings("ignore")

from woma.misc import glob_vars as gv
from woma.misc import utils
from woma.eos import eos
from woma.eos.T_rho import T_rho, set_T_rho_args


@njit
def L4_integrate(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    R2,
    R3,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    mat_id_L4,
    T_rho_type_id_L4,
    T_rho_args_L4,
):
    A1_r = np.linspace(R, 0, int(num_prof))
    A1_m_enc = np.zeros(A1_r.shape)
    A1_P = np.zeros(A1_r.shape)
    A1_T = np.zeros(A1_r.shape)
    A1_rho = np.zeros(A1_r.shape)
    A1_u = np.zeros(A1_r.shape)
    A1_mat_id = np.zeros(A1_r.shape)

    u_s = eos.u_rho_T(rho_s, T_s, mat_id_L4)
    T_rho_args_L4 = set_T_rho_args(
        T_s, rho_s, T_rho_type_id_L4, T_rho_args_L4, mat_id_L4
    )

    dr = A1_r[0] - A1_r[1]

    A1_m_enc[0] = M
    A1_P[0] = P_s
    A1_T[0] = T_s
    A1_rho[0] = rho_s
    A1_u[0] = u_s
    A1_mat_id[0] = mat_id_L4

    for i in range(1, A1_r.shape[0]):
        # Layer 4
        if A1_r[i] > R3:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L4
            T_rho_type_id = T_rho_type_id_L4
            T_rho_args = T_rho_args_L4
            rho0 = rho

        # Layer 3, 4 boundary
        elif A1_r[i] <= R3 and A1_r[i - 1] > R3:
            # New density, continuous temperature unless fixed entropy
            if T_rho_type_id_L3 == gv.type_ent:
                rho = eos.find_rho(
                    A1_P[i - 1],
                    mat_id_L3,
                    T_rho_type_id_L3,
                    T_rho_args_L3,
                    A1_rho[i - 1],
                    1e5,
                )
            else:
                rho = eos.rho_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L3)
                T_rho_args_L3 = set_T_rho_args(
                    A1_T[i - 1], rho, T_rho_type_id_L3, T_rho_args_L3, mat_id_L3
                )

            mat_id = mat_id_L3
            T_rho_type_id = T_rho_type_id_L3
            T_rho_args = T_rho_args_L3
            rho0 = A1_rho[i - 1]
        
        # Layer 3
        elif A1_r[i] > R2:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L3
            T_rho_type_id = T_rho_type_id_L3
            T_rho_args = T_rho_args_L3
            rho0 = rho

        # Layer 2, 3 boundary
        elif A1_r[i] <= R2 and A1_r[i - 1] > R2:
            # New density, continuous temperature unless fixed entropy
            if T_rho_type_id_L2 == gv.type_ent:
                rho = eos.find_rho(
                    A1_P[i - 1],
                    mat_id_L2,
                    T_rho_type_id_L2,
                    T_rho_args_L2,
                    A1_rho[i - 1],
                    1e5,
                )
            else:
                rho = eos.rho_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L2)
                T_rho_args_L2 = set_T_rho_args(
                    A1_T[i - 1], rho, T_rho_type_id_L2, T_rho_args_L2, mat_id_L2
                )

            mat_id = mat_id_L2
            T_rho_type_id = T_rho_type_id_L2
            T_rho_args = T_rho_args_L2
            rho0 = A1_rho[i - 1]
        
        # Layer 2
        elif A1_r[i] > R1:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L2
            T_rho_type_id = T_rho_type_id_L2
            T_rho_args = T_rho_args_L2
            rho0 = A1_rho[i - 1]

        # Layer 1, 2 boundary
        elif A1_r[i] <= R1 and A1_r[i - 1] > R1:
            # New density, continuous temperature unless fixed entropy
            if T_rho_type_id_L1 == gv.type_ent:
                rho = eos.find_rho(
                    A1_P[i - 1],
                    mat_id_L1,
                    T_rho_type_id_L1,
                    T_rho_args_L1,
                    A1_rho[i - 1],
                    1e5,
                )
            else:
                rho = eos.rho_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L1)
                T_rho_args_L1 = set_T_rho_args(
                    A1_T[i - 1], rho, T_rho_type_id_L1, T_rho_args_L1, mat_id_L1
                )

            mat_id = mat_id_L1
            T_rho_type_id = T_rho_type_id_L1
            T_rho_args = T_rho_args_L1
            rho0 = A1_rho[i - 1]

        # Layer 1
        elif A1_r[i] <= R1:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L1
            T_rho_type_id = T_rho_type_id_L1
            T_rho_args = T_rho_args_L1
            rho0 = A1_rho[i - 1]

        A1_m_enc[i] = A1_m_enc[i - 1] - 4 * np.pi * A1_r[i - 1] ** 2 * rho * dr
        A1_P[i] = A1_P[i - 1] + gv.G * A1_m_enc[i - 1] * rho / (A1_r[i - 1] ** 2) * dr
        A1_rho[i] = eos.find_rho(
            A1_P[i], mat_id, T_rho_type_id, T_rho_args, rho0, 1.1 * rho
        )
        A1_T[i] = T_rho(A1_rho[i], T_rho_type_id, T_rho_args, mat_id)
        A1_u[i] = eos.u_rho_T(A1_rho[i], A1_T[i], mat_id)
        A1_mat_id[i] = mat_id
        # Update the T-rho parameters
        if mat_id == gv.id_HM80_HHe and T_rho_type_id == gv.type_adb:
            T_rho_args = set_T_rho_args(
                A1_T[i], A1_rho[i], T_rho_type_id, T_rho_args, mat_id
            )

        if A1_m_enc[i] < 0:
            break

    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id


def L4_find_R1_given_M1_M_R2_R3_R(
    num_prof,
    R,
    R3,
    R2,
    M,
    M1,
    P_s,
    T_s,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    mat_id_L4,
    T_rho_type_id_L4,
    T_rho_args_L4,
    num_attempt=20,
    tol=0.01,
    verbosity=1,
    ):
        R1_min = 0.0
        R1_max = R2

        for i in range(num_attempt):

            R1_try = (R1_min + R1_max) * 0.5

            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L4_integrate(
                num_prof,
                R,
                M,
                P_s,
                T_s,
                rho_s,
                R1_try,
                R2,
                R3,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                mat_id_L4,
                T_rho_type_id_L4,
                T_rho_args_L4,
            )

            M1_try = A1_m_enc[A1_mat_id == mat_id_L1][0]

            # if A1_m_enc[-1] > 0.0:
            #     R1_min = R1_try
            # else:
            #     R1_max = R1_try

            if  M1_try > M1:
                R1_min = R1_try
            else:
                R1_max = R1_try

            tol_reached = (M1_try - M1) / M1
            #tol_reached = np.abs(R1_min - R1_max) / R1_max

            # Print progress
            if verbosity >= 1:
                print(
                    "\rR1 iterations: Iter %d(%d): R1=%.5gR_E: tol=%.2g(%.2g)"
                    % (i + 1, num_attempt, R1_try / gv.R_earth, tol_reached, tol),
                    end="  ",
                    flush=True,
                )

            # Error messages
            if np.abs(R1_try - R2) / R < 1 / (num_prof - 1):
                raise ValueError("R1 tends to R2. Please increase R2.")

            if R1_try / R < 1 / (num_prof - 1):
                raise ValueError("R1 tends to 0. Please decrease R1 or R.")

            if tol_reached < tol:
                if verbosity >= 1:
                    print("")
                break

        # Message if there is not convergence after num_attempt iterations
        if i == num_attempt - 1 and verbosity >= 1:
            print("\nR1 iterations: Warning: Convergence not reached after %d iterations." % (num_attempt))

        # Error messages
        if np.abs(R1_max - R2) / R2 < 2 * tol:
            raise ValueError("R1 tends to R2. Please increase R2.")

        if R1_max / R < 2 * tol:
            raise ValueError("R1 tends to 0. Please decrease R1 or R.")
        print('M1_try:', M1_try)
        return R1_try



def L4_find_R1_R2_given_M1_M2_M_R3_R(
    num_prof,
    R,
    R3,
    R2_min_guess,
    R2_max_guess,
    M,
    M1,
    M2,
    P_s,
    T_s,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    mat_id_L4,
    T_rho_type_id_L4,
    T_rho_args_L4,
    num_attempt=20,
    tol=0.01,
    verbosity=0,
    ):
        R2_min = R2_min_guess
        R2_max = R2_max_guess

        for i in range(num_attempt):
            R2_try = (R2_min + R2_max) * 0.5
        
            R1_try = L4_find_R1_given_M1_M_R2_R3_R(
                    num_prof,
                    R,
                    R3,
                    R2_try,
                    M,
                    M1,
                    P_s,
                    T_s,
                    rho_s,
                    mat_id_L1,
                    T_rho_type_id_L1,
                    T_rho_args_L1,
                    mat_id_L2,
                    T_rho_type_id_L2,
                    T_rho_args_L2,
                    mat_id_L3,
                    T_rho_type_id_L3,
                    T_rho_args_L3,
                    mat_id_L4,
                    T_rho_type_id_L4,
                    T_rho_args_L4,
                    num_attempt=20,
                    tol=0.01,
                    verbosity=1,
            )

            #print(R1_try)
            #assert False

            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L4_integrate(
                num_prof,
                R,
                M,
                P_s,
                T_s,
                rho_s,
                R1_try,
                R2_try,
                R3,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                mat_id_L4,
                T_rho_type_id_L4,
                T_rho_args_L4,
            )
            
            M2_try = A1_m_enc[A1_mat_id == mat_id_L2][0]

            # if A1_m_enc[-1] > 0.0:
            #     R2_min = R2_try
            # else:
            #     R2_max = R2_try

            if M2_try > M2:
                R2_min = R2_try
            else:
                R2_max = R2_try

            tol_reached = (M2_try - M2) / M2
            #tol_reached = np.abs(R2_min - R2_max) / R2_max

                # Print progress
            if verbosity >= 1:
                print(
                    "\rR2 iterations: Iter %d(%d): R2=%.5gR_E: tol=%.2g(%.2g)"
                    % (i + 1, num_attempt, R2_try / gv.R_earth, tol_reached, tol),
                    end="  ",
                    flush=True,
                )

            # Error messages
            if np.abs(R2_try - R1_try) / R < 1 / (num_prof - 1):
                raise ValueError("R2 tends to R1. Please decrease R1.")

            if np.abs(R2_try - R3) / R < 1 / (num_prof - 1):
                raise ValueError("R2 tends to R3. Please increase R3.")

            if tol_reached < tol:
                if verbosity >= 1:
                    print("")
                break

        # Message if there is not convergence after num_attempt iterations
        if i == num_attempt - 1 and verbosity >= 1:
            print("\nR2 iterations: Warning: Convergence not reached after %d iterations." % (num_attempt))

        # Error messages
        if np.abs(R2_max - R3) / R3 < 2 * tol:
            raise ValueError("R2 tends to R3. Please increase R3.")

        if np.abs(R2_min - R1_try) / R1_try < 2 * tol:
            raise ValueError("R2 tends to R1. Please decrease R1.")
        print(M2_try)
        return R1_try, R2_try


def L4_find_R1_R2_R3_given_M1_M2_M3_M_R(
        num_prof,
        R,
        R3_min_guess,
        R3_max_guess,
        R2_min_guess,
        R2_max_guess,
        M,
        M1,
        M2,
        M3,
        P_s,
        T_s,
        rho_s,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
        mat_id_L4,
        T_rho_type_id_L4,
        T_rho_args_L4,
        num_attempt=20,
        tol=0.01,
        verbosity=1,
    ):
        R3_min = R3_min_guess
        R3_max = R3_max_guess

        for i in range(num_attempt):
            R3_try = (R3_min + R3_max) * 0.5
        
            R1_try, R2_try = L4_find_R1_R2_given_M1_M2_M_R3_R(

                num_prof,
                R,
                R3_try,
                R2_min_guess,
                R2_max_guess,
                M,
                M1,
                M2,
                P_s,
                T_s,
                rho_s,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                mat_id_L4,
                T_rho_type_id_L4,
                T_rho_args_L4,
                num_attempt=20,
                tol=0.01,
                verbosity=1,
                )

            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L4_integrate(
                num_prof,
                R,
                M,
                P_s,
                T_s,
                rho_s,
                R1_try,
                R2_try,
                R3_try,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                mat_id_L4,
                T_rho_type_id_L4,
                T_rho_args_L4,
            )

            M3_try = A1_m_enc[A1_mat_id == mat_id_L3][0]

            if A1_m_enc[-1] > 0.0:
                R3_min = R3_try
            else:
                R3_max = R3_try

            tol_reached = np.abs(M3_try - M3) / M3

               # Print progress
            if verbosity >= 1:
                print(
                    "\rR3 iterations: Iter %d(%d): R2=%.5gR_E: tol=%.2g(%.2g)"
                    % (i + 1, num_attempt, R2_try / gv.R_earth, tol_reached, tol),
                    end="  ",
                    flush=True,
                )

            # Error messages
            if np.abs(R3_try - R2_try) / R < 1 / (num_prof - 1):
                raise ValueError("R3 tends to R2. Please decrease R2.")

            if np.abs(R3_try - R) / R < 1 / (num_prof - 1):
                raise ValueError("R3 tends to R. Please increase R.")

            if tol_reached < tol:
                if verbosity >= 1:
                    print("")
                break

        # Message if there is not convergence after num_attempt iterations
        if i == num_attempt - 1 and verbosity >= 1:
            print("\nR3 iterations: Warning: Convergence not reached after %d iterations." % (num_attempt))

        # Error messages
        if np.abs(R3_max - R) / R < 2 * tol:
            raise ValueError("R3 tends to R. Please increase R.")

        if np.abs(R3_min - R2_try) / R2_try < 2 * tol:
            raise ValueError("R3 tends to R2. Please decrease R2.")

        return R1_try, R2_try, R3_try


def L4_find_R1_R2_R3_R_given_M1_M2_M3_M(
        num_prof,
        R_min_guess,
        R_max_guess,
        R3_min_guess,
        R3_max_guess,
        R2_min_guess,
        R2_max_guess,
        M,
        M1,
        M2,
        M3,
        P_s,
        T_s,
        rho_s,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
        mat_id_L4,
        T_rho_type_id_L4,
        T_rho_args_L4,
        num_attempt=40,
        tol=0.01,
        verbosity=0,
    ):
        R_min = R_min_guess
        R_max = R_max_guess
        M123 = M1 + M2 + M3

        for i in range(num_attempt):
            R_try = (R_min + R_max) * 0.5
        
            R1_try, R2_try, R3_try = L4_find_R1_R2_R3_given_M1_M2_M3_M_R(

                num_prof,
                R_try,
                R3_min_guess,
                R3_max_guess,
                R2_min_guess,
                R2_max_guess,
                M,
                M1,
                M2,
                M3,
                P_s,
                T_s,
                rho_s,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                mat_id_L4,
                T_rho_type_id_L4,
                T_rho_args_L4,
                num_attempt=40,
                tol=0.01,
                verbosity=0,
                )

            A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L4_integrate(
                num_prof,
                R_try,
                M,
                P_s,
                T_s,
                rho_s,
                R1_try,
                R2_try,
                R3_try,
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                mat_id_L4,
                T_rho_type_id_L4,
                T_rho_args_L4,
            )

            M123_try = A1_m_enc[A1_mat_id != mat_id_L4][0]

            if M123_try > M123:
                R_min = R_try
            else:
                R_max = R_try

            tol_reached = np.abs(M123_try - M123) / M123

               # Print progress
            if verbosity >= 1:
                print(
                    "\rIter %d(%d): R2=%.5gR_E: tol=%.2g(%.2g)"
                    % (i + 1, num_attempt, R2_try / gv.R_earth, tol_reached, tol),
                    end="  ",
                    flush=True,
                )

            # Error messages
            if np.abs(R_try - R3_try) / R < 1 / (num_prof - 1):
                raise ValueError("R tends to R3. Please decrease R3.")

            if np.abs(R_try - R_min_guess) / R < 1 / (num_prof - 1):
                raise ValueError("R tends to R_min_guess. Please increase R.")

            if tol_reached < tol:
                if verbosity >= 1:
                    print("")
                break

        # Message if there is not convergence after num_attempt iterations
        if i == num_attempt - 1 and verbosity >= 1:
            print("\nWarning: Convergence not reached after %d iterations." % (num_attempt))

        # Error messages
        if np.abs(R_try - R_max_guess) / R_try < 2 * tol:
            raise ValueError("R tends to R_max_guess. Please increase R_max_guess.")

        if np.abs(R_try - R_min_guess) / R_try < 2 * tol:
            raise ValueError("R tends to R_min_guess. Please decrease R_min_guess.")

        return R1_try, R2_try, R3_try, R_try


