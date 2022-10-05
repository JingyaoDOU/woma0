""" 
WoMa equations of state (EoS)
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from woma.eos import hm80, idg, sesame, tillotson
from woma.eos.T_rho import T_rho
from woma.misc import glob_vars as gv


# ========
# Generic
# ========
@njit
def Z_rho_T(rho, T, mat_id, Z_choice):
    """Compute an equation of state parameter from the density and temperature,
    for any EoS.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Z_choice : str
        The parameter to calculate, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag.

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        return sesame.Z_rho_T(rho, T, mat_id, Z_choice)
    else:
        raise ValueError("Not yet implemented for this EoS")


@njit
def A1_Z_rho_T(A1_rho, A1_T, A1_mat_id, Z_choice):
    """Compute equation of state parameters from arrays of density and
    temperature, for any EoS.

    Parameters
    ----------
    A1_rho : [float]
        Densities (kg m^-3).

    A1_T : float
        Temperatures (K).

    A1_mat_id : [int]
        Material ids.

    Z_choice : str
        The parameter to calculate, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag.

    Returns
    -------
    A1_Z : float
        The chosen parameter values (SI).
    """

    assert A1_rho.ndim == 1
    assert A1_T.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_rho.shape[0] == A1_T.shape[0]
    assert A1_rho.shape[0] == A1_mat_id.shape[0]

    A1_Z = np.zeros_like(A1_rho)

    for i, rho in enumerate(A1_rho):
        A1_Z[i] = Z_rho_T(A1_rho[i], A1_T[i], A1_mat_id[i], Z_choice)

    return A1_Z


@njit
def Z_rho_Y(rho, Y, mat_id, Z_choice, Y_choice):
    """Compute an equation of state parameter from the density and another
    parameter, for any EoS..

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    Y : float
        The chosen input parameter (SI).

    mat_id : int
        Material id.

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        return sesame.Z_rho_Y(rho, Y, mat_id, Z_choice, Y_choice)
    else:
        raise ValueError("Not yet implemented for this EoS")


@njit
def A1_Z_rho_Y(A1_rho, A1_Y, A1_mat_id, Z_choice, Y_choice):
    """Compute equation of state parameters from arrays of density and
    another parameter, for any EoS.

    Parameters
    ----------
    A1_rho : [float]
        Densities (kg m^-3).

    A1_Y : float
        The chosen input parameter values (SI).

    A1_mat_id : [int]
        Material ids.

    Z_choice, Y_choice : str
        The parameter to calculate, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    A1_Z : float
        The chosen parameter values (SI).
    """
    assert A1_rho.ndim == 1
    assert A1_Y.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_rho.shape[0] == A1_Y.shape[0]
    assert A1_rho.shape[0] == A1_mat_id.shape[0]

    A1_Z = np.zeros_like(A1_rho)

    for i, rho in enumerate(A1_rho):
        A1_Z[i] = Z_rho_Y(A1_rho[i], A1_Y[i], A1_mat_id[i], Z_choice, Y_choice)

    return A1_Z


@njit
def Z_X_T(X, T, mat_id, Z_choice, X_choice):
    """Compute an equation of state parameter from another parameter and the
    temperature, for any EoS.

    Parameters
    ----------
    X : float
        The chosen input parameter (SI).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Z_choice, X_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        return sesame.Z_X_T(X, T, mat_id, Z_choice, X_choice)
    else:
        raise ValueError("Not yet implemented for this EoS")


@njit
def A1_Z_X_T(A1_X, A1_T, A1_mat_id, Z_choice, X_choice):
    """Compute equation of state parameters from arrays of density and
    another parameter, for any EoS.

    Parameters
    ----------
    A1_X : [float]
        The chosen input parameter values (SI).

    A1_T : [float]
        Temperatures (K).

    A1_mat_id : [int]
        Material ids.

    Z_choice, X_choice : str
        The parameter to calculate, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    A1_Z : float
        The chosen parameter values (SI).
    """
    assert A1_X.ndim == 1
    assert A1_T.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_X.shape[0] == A1_T.shape[0]
    assert A1_X.shape[0] == A1_mat_id.shape[0]

    A1_Z = np.zeros_like(A1_X)

    for i, X in enumerate(A1_X):
        A1_Z[i] = Z_X_T(A1_X[i], A1_T[i], A1_mat_id[i], Z_choice, X_choice)

    return A1_Z


# ========
# Pressure
# ========
@njit
def P_u_rho(u, rho, mat_id):
    """Compute the pressure from the specific internal energy
    and density, for any EoS.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    P : float
        Pressure (Pa).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        P = idg.P_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_Til:
        P = tillotson.P_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_HM80:
        P = hm80.P_u_rho(u, rho, mat_id)
        if np.isnan(P):
            P = 0.0
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        P = sesame.P_u_rho(u, rho, mat_id)
        if np.isnan(P):
            P = 0.0
    else:
        raise ValueError("Invalid material ID")
    return P


@njit
def A1_P_u_rho(A1_u, A1_rho, A1_mat_id):
    """Compute the pressures from arrays of specific internal energy and
    density, for any EoS.

    Parameters
    ----------
    A1_u : [float]
        Specific internal energy (J kg^-1).

    A1_rho : [float]
        Density (kg m^-3).

    A1_mat_id : [int]
        Material id.

    Returns
    -------
    A1_P : float
        Pressure (Pa).
    """

    assert A1_u.ndim == 1
    assert A1_rho.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_u.shape[0] == A1_rho.shape[0]
    assert A1_u.shape[0] == A1_mat_id.shape[0]

    A1_P = np.zeros_like(A1_u)

    for i, u in enumerate(A1_u):
        A1_P[i] = P_u_rho(A1_u[i], A1_rho[i], A1_mat_id[i])

    return A1_P


@njit
def P_T_rho(T, rho, mat_id):
    """Compute the pressure from the temperature and density, for any EoS.

    Parameters
    ----------
    T : float
        Temperature (K).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    P : float
        Pressure (Pa).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        P = idg.P_T_rho(T, rho, mat_id)
    elif mat_type == gv.type_Til:
        P = tillotson.P_T_rho(T, rho, mat_id)
    elif mat_type == gv.type_HM80:
        P = hm80.P_T_rho(T, rho, mat_id)
        if np.isnan(P):
            P = 0.0
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        P = sesame.P_T_rho(T, rho, mat_id)
        if np.isnan(P):
            P = 0.0
    else:
        raise ValueError("Invalid material ID")
    return P


@njit
def A1_P_T_rho(A1_T, A1_rho, A1_mat_id):
    """Compute the pressures from arrays of temperature and density, for any
    EoS.

    Parameters
    ----------
    A1_T : [float]
        Temperature (K).

    A1_rho : [float]
        Density (kg m^-3).

    A1_mat_id : [int]
        Material id.

    Returns
    -------
    A1_P : [float]
        Pressure (Pa).
    """
    assert A1_T.ndim == 1
    assert A1_rho.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_T.shape[0] == A1_rho.shape[0]
    assert A1_T.shape[0] == A1_mat_id.shape[0]

    A1_P = np.zeros_like(A1_T)

    for i, T in enumerate(A1_T):
        A1_P[i] = P_T_rho(A1_T[i], A1_rho[i], A1_mat_id[i])

    return A1_P


# ========
# Temperature
# ========
@njit
def T_u_rho(u, rho, mat_id):
    """Compute the pressure from the density and temperature, for any EoS.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    T : float
        Temperature (K).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        T = idg.T_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_Til:
        T = tillotson.T_u_rho(u, rho, mat_id)
    elif mat_type == gv.type_HM80:
        T = hm80.T_u_rho(u, rho, mat_id)
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        T = sesame.T_u_rho(u, rho, mat_id)
    else:
        raise ValueError("Invalid material ID")
    return T


@njit
def A1_T_u_rho(A1_u, A1_rho, A1_mat_id):
    """Compute the pressures from arrays of density and temperature, for any
    EoS.

    Parameters
    ----------
    A1_u : [float]
        Specific internal energy (J kg^-1).

    A1_rho : [float]
        Density (kg m^-3).

    A1_mat_id : [int]
        Material id.

    Returns
    -------
    A1_T : [float]
        Temperature (K).
    """

    assert A1_u.ndim == 1
    assert A1_rho.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_u.shape[0] == A1_rho.shape[0]
    assert A1_u.shape[0] == A1_mat_id.shape[0]

    A1_T = np.zeros_like(A1_u)

    for i, u in enumerate(A1_u):
        A1_T[i] = T_u_rho(A1_u[i], A1_rho[i], A1_mat_id[i])

    return A1_T


# ========
# Specific internal energy
# ========
@njit
def u_rho_T(rho, T, mat_id):
    """Compute the specific internal energy from the density and temperature, for any EoS.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    u : float
        Specific internal energy (J kg^-1).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        u = idg.u_rho_T(rho, T, mat_id)
    elif mat_type == gv.type_Til:
        u = tillotson.u_rho_T(rho, T, mat_id)
    elif mat_type == gv.type_HM80:
        u = hm80.u_rho_T(rho, T, mat_id)
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        u = sesame.u_rho_T(rho, T, mat_id)
    else:
        raise ValueError("Invalid material ID")
    return u


@njit
def A1_u_rho_T(A1_rho, A1_T, A1_mat_id):
    """Compute the specific internal energies from arrays of density and
    temperature, for any EoS.

    Parameters
    ----------
    A1_rho : [float]
        Density (kg m^-3).

    A1_T : [float]
        Temperature (K).

    A1_mat_id : [int]
        Material id.

    Returns
    -------
    A1_u : [float]
        Specific internal energy (J kg^-1).
    """
    assert A1_rho.ndim == 1
    assert A1_T.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_rho.shape[0] == A1_T.shape[0]
    assert A1_rho.shape[0] == A1_mat_id.shape[0]

    A1_u = np.zeros_like(A1_rho)

    for i, rho in enumerate(A1_rho):
        A1_u[i] = u_rho_T(A1_rho[i], A1_T[i], A1_mat_id[i])

    return A1_u


# ========
# Specific entropy
# ========
@njit
def s_rho_T(rho, T, mat_id):
    """Compute the specific entropy from the density and temperature, for any EoS.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    s : float
        Specific entropy (J kg^-1 K^-1).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        s = sesame.s_rho_T(rho, T, mat_id)
    else:
        raise ValueError("Entropy not implemented for this material type.")
    return s


@njit
def A1_s_rho_T(A1_rho, A1_T, A1_mat_id):
    """Compute the specific entropies from arrays of density and temperature,
    for any EoS.

    Parameters
    ----------
    A1_rho : [float]
        Density (kg m^-3).

    A1_T : [float]
        Temperature (K).

    A1_mat_id : [int]
        Material id.

    Returns
    -------
    A1_s : [float]
        Specific entropy (J kg^-1 K^-1).
    """
    assert A1_T.ndim == 1
    assert A1_rho.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_T.shape[0] == A1_rho.shape[0]
    assert A1_T.shape[0] == A1_mat_id.shape[0]

    A1_s = np.zeros_like(A1_T)

    for i, rho in enumerate(A1_rho):
        A1_s[i] = s_rho_T(A1_rho[i], A1_T[i], A1_mat_id[i])

    return A1_s


@njit
def s_u_rho(u, rho, mat_id):
    """Compute the specific entropy from the specific internal energy and
    density, for any EoS.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    s : float
        Specific entropy (J kg^-1 K^-1).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        s = sesame.s_u_rho(u, rho, mat_id)
    else:
        raise ValueError("Entropy not implemented for this material type.")
    return s


@njit
def A1_s_u_rho(A1_u, A1_rho, A1_mat_id):
    """Compute the specific entropies from arrays of specific internal energy
    and density, for any EoS.

    Parameters
    ----------
    A1_u : [float]
        Specific internal energy (J kg^-1).

    A1_rho : [float]
        Density (kg m^-3).

    A1_mat_id : [int]
        Material id.

    Returns
    -------
    A1_s : [float]
        Specific entropy (J kg^-1 K^-1).
    """

    assert A1_u.ndim == 1
    assert A1_rho.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_u.shape[0] == A1_rho.shape[0]
    assert A1_u.shape[0] == A1_mat_id.shape[0]

    A1_s = np.zeros_like(A1_u)

    for i, u in enumerate(A1_u):
        A1_s[i] = s_u_rho(A1_u[i], A1_rho[i], A1_mat_id[i])

    return A1_s


# ========
# Density
# ========
@njit
def find_rho(P_des, mat_id, T_rho_type, T_rho_args, rho_min, rho_max):
    """Find the density that satisfies P(u(rho), rho) = P_des, for any EoS.

    Parameters
    ----------
    P_des : float
        The required pressure (Pa).

    mat_id : int
        Material id.

    T_rho_type : int
        Relation between T and rho to be used.

    T_rho_args : [?]
        Extra arguments to determine the relation.

    rho_min : float
        Lower bound for where to look the root (kg m^-3).

    rho_max : float
        Upper bound for where to look the root (kg m^-3).

    Returns
    -------
    rho_mid : float
        The density which satisfies P(u(rho), rho) = P_des (kg m^-3).
    """

    assert rho_min > 0
    assert P_des >= 0
    assert rho_min < rho_max

    tolerance = 1e-5

    T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
    P_min = P_T_rho(T_min, rho_min, mat_id)
    T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
    P_max = P_T_rho(T_max, rho_max, mat_id)
    rho_mid = (rho_min + rho_max) / 2.0
    T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
    P_mid = P_T_rho(T_mid, rho_mid, mat_id)
    rho_aux = rho_min + 1e-6
    T_aux = T_rho(rho_aux, T_rho_type, T_rho_args, mat_id)
    P_aux = P_T_rho(T_aux, rho_aux, mat_id)

    if (P_min < P_des < P_max) or (P_min > P_des > P_max):
        max_counter = 200
        counter = 0
        while np.abs(rho_max - rho_min) > tolerance and counter < max_counter:
            T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
            P_min = P_T_rho(T_min, rho_min, mat_id)
            T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
            P_max = P_T_rho(T_max, rho_max, mat_id)
            T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
            P_mid = P_T_rho(T_mid, rho_mid, mat_id)

            f0 = P_des - P_min
            f2 = P_des - P_mid

            if f0 * f2 > 0:
                rho_min = rho_mid
            else:
                rho_max = rho_mid

            rho_mid = (rho_min + rho_max) / 2.0
            counter += 1

        return rho_mid

    elif P_min == P_des == P_aux != P_max and P_min < P_max:
        while np.abs(rho_max - rho_min) > tolerance:
            rho_mid = (rho_min + rho_max) / 2.0
            T_min = T_rho(rho_min, T_rho_type, T_rho_args, mat_id)
            P_min = P_T_rho(T_min, rho_min, mat_id)
            T_max = T_rho(rho_max, T_rho_type, T_rho_args, mat_id)
            P_max = P_T_rho(T_max, rho_max, mat_id)
            rho_mid = (rho_min + rho_max) / 2.0
            T_mid = T_rho(rho_mid, T_rho_type, T_rho_args, mat_id)
            P_mid = P_T_rho(T_mid, rho_mid, mat_id)

            if P_mid == P_des:
                rho_min = rho_mid
            else:
                rho_max = rho_mid

            rho_mid = rho_mid = (rho_min + rho_max) / 2.0

        return rho_mid

    elif P_des < P_min < P_max:
        return find_rho(P_des, mat_id, T_rho_type, T_rho_args, rho_min / 2, rho_max)
    elif P_des > P_max > P_min:
        return find_rho(P_des, mat_id, T_rho_type, T_rho_args, rho_min, 2 * rho_max)
    elif P_des > P_min > P_max:
        return rho_min
    elif P_des < P_max < P_min:
        return rho_max
    else:
        # For debugging
        # print(P_des)
        # print(mat_id)
        # print(T_rho_type, T_rho_args)
        # print(rho_min, rho_max)
        # print(T_min, T_max)
        # print(P_min, P_max)
        e = "Critical error in find_rho."
        raise ValueError(e)


@njit
def rho_P_T(P, T, mat_id):
    """Compute the density from the pressure and temperature, for any EoS.

    Parameters
    ----------
    P : float
        Pressure (Pa).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    rho : float
        Density (kg m^-3).
    """
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        assert T > 0
        rho_min = 1e-10
        rho_max = 1e5
    elif mat_type == gv.type_Til:
        rho_min = 1e-7
        rho_max = 1e6
    elif mat_type == gv.type_HM80:
        assert T > 0
        if mat_id == gv.id_HM80_HHe:
            rho_min = 1e-1
            rho_max = 1e5
        elif mat_id == gv.id_HM80_ice:
            rho_min = 1e0
            rho_max = 1e5
        elif mat_id == gv.id_HM80_rock:
            rho_min = 1e0
            rho_max = 40000
    elif mat_type in [gv.type_SESAME, gv.type_ANEOS]:
        assert T > 0
        assert P > 0

        rho_min = 1e-9
        rho_max = 1e5
    else:
        raise ValueError("Invalid material ID")

    return find_rho(P, mat_id, 1, [float(T), 0.0], rho_min, rho_max)


@njit
def A1_rho_P_T(A1_P, A1_T, A1_mat_id):
    """Compute the densities from arrays of pressure and temperature, for any
    EoS.

    Parameters
    ----------
    A1_P : [float]
        Pressure (Pa).

    A1_T : [float]
        Temperature (K).

    A1_mat_id : [int]
        Material id.

    Returns
    -------
    A1_rho : [float]
        Density (kg m^-3).
    """
    assert A1_P.ndim == 1
    assert A1_T.ndim == 1
    assert A1_mat_id.ndim == 1
    assert A1_P.shape[0] == A1_T.shape[0]
    assert A1_P.shape[0] == A1_mat_id.shape[0]

    A1_rho = np.zeros_like(A1_P)

    for i, P in enumerate(A1_P):
        A1_rho[i] = rho_P_T(A1_P[i], A1_T[i], A1_mat_id[i])

    return A1_rho


# ========
# Derived functions
# ========


# ========
# Misc
# ========
def plot_EoS_P_rho_fixed_T(
    mat_id_1, mat_id_2, T, P_min=0.1, P_max=1e11, rho_min=100, rho_max=15000
):
    """Plot the EoS pressure as a function of density for various temperatures.

    Parameters
    ----------
    mat_id_1 : int
        Material id for the first material.

    mat_id_2 : int
        Material id for the second material.

    T : float
        Fixed temperature (K).

    P_min : float
        Minimum pressure (Pa) to consider.

    P_max : float
        Maximum pressure (Pa) to consider.

    rho_min : float
        Minimum density (kg m^-3) to consider.

    rho_min : float
        Maximum density (kg m^-3) to consider.
    """

    rho = np.linspace(rho_min, rho_max, 1000)
    P_1 = np.zeros_like(rho)
    P_2 = np.zeros_like(rho)
    for i, rho_i in enumerate(rho):
        P_1[i] = P_T_rho(T, rho_i, mat_id_1)
        P_2[i] = P_T_rho(T, rho_i, mat_id_2)

    plt.figure()
    plt.scatter(rho, P_1, label=str(mat_id_1))
    plt.scatter(rho, P_2, label=str(mat_id_2))
    plt.legend(title="Material")
    plt.xlabel(r"$\rho$ [kg m$^{-3}$]")
    plt.ylabel(r"$P$ [Pa]")
    plt.show()


def plot_table_SESAME(mat, Z_choice, A1_fig_ax=None):
    """Plot all entries in a SESAME or ANEOS etc (rho, T) table.
    Parameters
    ----------
    mat : str
        The material name.
    Z_choice : str
        The table parameter to colour by, choose from:
            "P"     Pressure.
            "u"     Specific internal energy.
            "s"     Specific entropy.
    A1_fig_ax : [fig, ax] (opt.)
        If provided, then plot on this existing figure and axes instead of
        making new ones.
    Returns
    -------
    A1_fig_ax : [fig, ax]
        The figure and axes.
    """
    mat_id = gv.Di_mat_id[mat]

    # Load tables
    ut.load_eos_tables(mat)

    # Load table data
    if mat_id == gv.id_SESAME_iron:
        A1_log_T = sesame.A1_log_T_SESAME_iron
        A1_log_rho = sesame.A1_log_rho_SESAME_iron
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SESAME_iron
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SESAME_iron
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SESAME_iron
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_T = sesame.A1_log_T_SESAME_basalt
        A1_log_rho = sesame.A1_log_rho_SESAME_basalt
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SESAME_basalt
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SESAME_basalt
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SESAME_basalt
    elif mat_id == gv.id_SESAME_water:
        A1_log_T = sesame.A1_log_T_SESAME_water
        A1_log_rho = sesame.A1_log_rho_SESAME_water
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SESAME_water
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SESAME_water
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SESAME_water
    elif mat_id == gv.id_SS08_water:
        A1_log_T = sesame.A1_log_T_SS08_water
        A1_log_rho = sesame.A1_log_rho_SS08_water
        if Z_choice == "P":
            A2_Z = sesame.A2_P_SS08_water
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_SS08_water
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_SS08_water
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_T = sesame.A1_log_T_ANEOS_forsterite
        A1_log_rho = sesame.A1_log_rho_ANEOS_forsterite
        if Z_choice == "P":
            A2_Z = sesame.A2_P_ANEOS_forsterite
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_ANEOS_forsterite
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_ANEOS_forsterite
        # elif Z_choice == "phase":
        #     A2_Z = sesame.A2_phase_ANEOS_forsterite
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_T = sesame.A1_log_T_ANEOS_iron
        A1_log_rho = sesame.A1_log_rho_ANEOS_iron
        if Z_choice == "P":
            A2_Z = sesame.A2_P_ANEOS_iron
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_ANEOS_iron
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_ANEOS_iron
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_T = sesame.A1_log_T_ANEOS_Fe85Si15
        A1_log_rho = sesame.A1_log_rho_ANEOS_Fe85Si15
        if Z_choice == "P":
            A2_Z = sesame.A2_P_ANEOS_Fe85Si15
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_ANEOS_Fe85Si15
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_ANEOS_Fe85Si15
    elif mat_id == gv.id_AQUA:
        A1_log_T = sesame.A1_log_T_AQUA
        A1_log_rho = sesame.A1_log_rho_AQUA
        if Z_choice == "P":
            A2_Z = sesame.A2_P_AQUA
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_AQUA
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_AQUA
    elif mat_id == gv.id_CMS19_H:
        A1_log_T = sesame.A1_log_T_CMS19_H
        A1_log_rho = sesame.A1_log_rho_CMS19_H
        if Z_choice == "P":
            A2_Z = sesame.A2_P_CMS19_H
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_CMS19_H
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_CMS19_H
    elif mat_id == gv.id_CMS19_He:
        A1_log_T = sesame.A1_log_T_CMS19_He
        A1_log_rho = sesame.A1_log_rho_CMS19_He
        if Z_choice == "P":
            A2_Z = sesame.A2_P_CMS19_He
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_CMS19_He
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_CMS19_He
    elif mat_id == gv.id_CMS19_HHe:
        A1_log_T = sesame.A1_log_T_CMS19_HHe
        A1_log_rho = sesame.A1_log_rho_CMS19_HHe
        if Z_choice == "P":
            A2_Z = sesame.A2_P_CMS19_HHe
        elif Z_choice == "u":
            A2_Z = sesame.A2_u_CMS19_HHe
        elif Z_choice == "s":
            A2_Z = sesame.A2_s_CMS19_HHe
    else:
        raise ValueError("Invalid material ID")
    A1_T = np.exp(A1_log_T)
    A1_rho = np.exp(A1_log_rho)
    num_T = len(A1_T)
    num_rho = len(A1_rho)

    # Skip if no data (e.g. entropies all zero)
    if np.all(A2_Z == 0):
        print("%s %s all zero" % (mat, Z_choice))
        return None, None

    # Colour parameter
    cmap = plt.get_cmap("viridis")
    vmin = np.nanmin(A2_Z[A2_Z > 0])
    vmax = np.nanmax(A2_Z[A2_Z < np.inf])
    norm = mpl.colors.LogNorm()

    # Figure
    if A1_fig_ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.gca()
    else:
        fig, ax = A1_fig_ax
        plt.figure(fig.number)

    # Roughly adjust marker size by number of points
    s_def = 3**2
    num_def = 200
    s = min(s_def, (np.sqrt(s_def) * num_def / max(num_rho, num_T)) ** 2)

    # Plot each row
    for i_T, T in enumerate(A1_T):
        scat = ax.scatter(
            A1_rho,
            np.full(num_rho, T),
            marker=".",
            s=s,
            c=A2_Z[:, i_T],
            edgecolor="none",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )

        # Plot non-positive values in grey
        A1_sel_zero = np.where(A2_Z[:, i_T] <= 0)[0]
        ax.scatter(
            A1_rho[A1_sel_zero],
            np.full(num_rho, T)[A1_sel_zero],
            marker=".",
            s=s,
            c="0.4",
            edgecolor="none",
        )

    # Colour bar
    cbar = plt.colorbar(scat)
    if Z_choice == "P":
        cbar.set_label(r"Pressure (Pa)")
    elif Z_choice == "u":
        cbar.set_label(r"Specific internal energy (J kg$^{-1}$)")
    elif Z_choice == "s":
        cbar.set_label(r"Specific entropy (J K$^{-1}$ kg$^{-1}$)")

    # Axes etc
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density ($\rm{kg m}^{-3}$)")
    ax.set_ylabel(r"Temperature (K)")
    ax.set_title(r"%s $%s(\rho, T)$" % (mat, Z_choice))

    plt.tight_layout()

    return fig, ax
