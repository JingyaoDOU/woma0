""" 
WoMa ideal gas equations of state
"""

from numba import njit

from woma.misc import glob_vars as gv


@njit
def idg_gamma(mat_id):
    """Return the adiabatic index gamma for an ideal gas.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    gamma : float
        Adiabatic index.
    """
    if mat_id == gv.id_idg_HHe:
<<<<<<< HEAD
        return 5/3   # Previous it's 1.4 the value of air
=======
        return 5/3
>>>>>>> jy_dev
    elif mat_id == gv.id_idg_N2:
        return 1.4
    elif mat_id == gv.id_idg_CO2:
        return 1.29
    else:
        raise ValueError("Invalid material ID")


@njit
def P_u_rho(u, rho, mat_id):
    """Compute the pressure from the internal energy and density.

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
    # Adiabatic constant
    gamma = idg_gamma(mat_id)

    P = (gamma - 1) * u * rho

    return P


@njit
def C_V_idg(mat_id):
    """Return the specific heat capacity.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    C_V : float
        Specific heat capacity (J kg^-1 K^-1).
    """
    if mat_id == gv.id_idg_HHe:
        return 9093.98
    elif mat_id == gv.id_idg_N2:
        return 742.36
    elif mat_id == gv.id_idg_CO2:
        return 661.38
    else:
        raise ValueError("Invalid material ID")


@njit
def u_rho_T(rho, T, mat_id):
    """Compute the internal energy from the density and temperature.

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
        return C_V_idg(mat_id) * T
    else:
        raise ValueError("Invalid material ID")


@njit
def P_T_rho(T, rho, mat_id):
    """Compute the pressure from the density and temperature.

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
    u : float
        Specific internal energy (J kg^-1).
    """

    mat_type = mat_id // gv.type_factor

    if mat_type == gv.type_idg:

        u = u_rho_T(rho, T, mat_id)
        P = P_u_rho(u, rho, mat_id)

    else:
        raise ValueError("Invalid material ID")

    return P


@njit
def T_u_rho(u, rho, mat_id):
    """Compute the pressure from the density and temperature.

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
<<<<<<< HEAD

    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_idg:
        print('T from Cv')
        return T_rho(rho, mat_id) 
    else:
        raise ValueError("Invalid material ID")

    #raise ValueError("T_u_rho function not implemented for ideal gas.")
    #return 0.0


@njit
def u_P_rho(P, rho, mat_id):
    """Compute the internal energy from the pressure and density

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    P: float
        Pressure  (Pa)


    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    T : float
        Temperature (K).
    """

    gamma = idg_gamma(mat_id)
    return (1/(gamma-1))*P/rho
    
    #raise ValueError("T_u_rho function not implemented for ideal gas.")
    #return 0.0

@njit
def P_u_rho(u, rho, mat_id):
    """Compute the pressure from the density and internal energy.

    Parameters
    ----------
    u : float
        Specific Internal Energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    P : float
        Pressure (Pa).
    """

    gamma = idg_gamma(mat_id)

    return (gamma-1)*u*rho

@njit
def s_P_rho(P, rho, mat_id):
    """Compute the specific entropy from the density and internal energy.

    """

    minus_gamma = -idg_gamma(mat_id)

    return P * rho**(minus_gamma)

@njit
def T_rho(rho, mat_id):
    """Compute the temperature from the density.

    """

    gamma = idg_gamma(mat_id)

    return (rho**(gamma-1))/(gamma-1)

@njit
def u_s_rho(s,rho, mat_id):
    """Compute the temperature from the density.

    """

    gamma = idg_gamma(mat_id)

    return s * (rho**(gamma-1))/(gamma-1)
=======
    raise ValueError("T_u_rho function not implemented for ideal gas.")
    return 0.0

@njit
def s_u_rho(u, rho, mat_id):
    """Compute the entropy from the internal energy and density.

    """

    gamma = idg_gamma(mat_id)

    return (gamma-1) * u * rho**(1-gamma)

@njit
def u_rho_P(rho, P, mat_id):
    """Compute the internal energy from the density and pressure.

    """
    gamma = idg_gamma(mat_id)

    return P / (rho * (gamma-1))
>>>>>>> jy_dev
