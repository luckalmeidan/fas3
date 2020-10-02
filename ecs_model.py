import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root as solver
from scipy.optimize import minimize

import CoolProp
from CoolProp.CoolProp import PropsSI
from CoolProp.Plots import PropertyPlot

cfm_2_m3s = 1 / 2119
lbmin_2_kgs = 1 / 132.277
ft_2_m = 1 / 3.282
psi2pa = 6894.76
kpa2bar = 0.01
kW2btumin = 3412.142
# Constants
g = 9.81
R = 287.05

Cp = 1.004


def atmosferaISA(H, deltaT):
    """

    :param H:
    :param deltaT:
    :return:
    """
    R = 287
    g = 9.80665
    T0 = 288.15
    Th = -6.5e-3
    P0 = 101325
    rho0 = 1.225

    if H <= 11000:
        T = T0 + Th * H
        P = P0 * (1 + ((Th * H) / T0)) ** (-g / (R * Th))
        rho = rho0 * (1 + ((Th * H) / T0)) ** ((-g / (R * Th)) - 1)

    elif H <= 20000:

        rho11, P11, T = atmosferaISA(11000, 0)
        P = P11 * np.exp(-g * (H - 11000) / (R * T))
        rho = P / (R * T)

    else:

        The = 0.001
        rho20, P20, T20 = atmosferaISA(20000, 0)
        T = T20 + The * (H - 20000)
        P = P20 * (1 + The * (H - 20000) / T20) ** (-g / (R * The))
        rho = P / (R * T)

    T = T + deltaT
    P = P
    rho = P / (R * T)

    return rho, P, T


def ram_compression(m, T_in, P_in, mach, gamma, eta):
    """

    :param m:
    :param T_in:
    :param P_in:
    :param mach:
    :param gamma:
    :param eta:
    :return:
    """
    T_out = T_in * (1 + (gamma - 1) / 2 * mach ** 2)
    P_out = eta * P_in * (1 + (gamma - 1) / 2 * mach ** 2) ** (gamma / (gamma - 1))
    P_out = P_in * (1 + eta * (T_out / T_in - 1)) ** (gamma / (gamma - 1))

    work = m * Cp * (T_out - T_in)

    return T_out, P_out, work


def compressor(m, T_in, P_in, r, gamma, eta):
    """

    :param m:
    :param T_in:
    :param P_in:
    :param r:
    :param gamma:
    :param eta:
    :return:
    """
    T_out = T_in * (1 + 1 / eta * (r ** ((gamma - 1) / gamma) - 1))
    P_out = P_in * r
    work = m * Cp * (T_out - T_in)
    return T_out, P_out, work


def heat_exchanger(m, T_in, T_cooling, hx_eff):
    """

    :param m:
    :param T_in:
    :param T_cooling:
    :param hx_eff:
    :return:
    """
    T_out = T_in - hx_eff * (T_in - T_cooling)
    Q = m * Cp * (T_in - T_out)
    return T_out, Q


def turbine(m, T_in, P_in, r, gamma, eta):
    """

    :param m:
    :param T_in:
    :param P_in:
    :param r:
    :param gamma:
    :param eta:
    :return:
    """
    P_out = P_in * r
    T_out = T_in * (eta * ((P_out / P_in) ** ((gamma - 1) / gamma) - 1) + 1)
    work = m * Cp * (T_in - T_out)

    return T_out, P_out, work


def get_residual(m, T2, P4b, T4b, P5, P7, gamma, hx_eff, eta_c, eta_t, alpha):
    _, _, _, _, w_t, w_sc = loop(m, T2, P4b, T4b, P5, P7, gamma, hx_eff, eta_c, eta_t)

    residual = (alpha * w_t - w_sc)
    return residual


def loop(m, T2, P4b, T4b, P5, P7, gamma, hx_eff, eta_c, eta_t):
    r_c = P5 / P4b

    T5, P5t, w_sc = compressor(1, T4b, P4b, r_c, gamma, eta_c)

    P6 = P5
    T6, Qshx = heat_exchanger(m, T5, T2, hx_eff)

    r_t = P7 / P6

    T7, P7t, w_t = turbine(m, T6, P5, r_t, gamma, eta_t)

    return T5, T6, T7, Qshx, w_t, w_sc


def simple_bootstrap_cycle(mach, m, e_recirc, H_cab_ft, Tcab, n_pax, Qrequired, alpha, print_results=False):
    """

    :param mach:
    :param m:
    :param H_cab_ft:
    :param Tcab:
    :return:
    """

    # Flight Conditions
    H_ft = 38000
    H_m = H_ft * ft_2_m

    H_cab_m = H_cab_ft * ft_2_m
    H_m = H_ft * ft_2_m
    deltaT = 0
    Tcab = 24 + 273
    rho, Pcab, T = atmosferaISA(H_cab_m, deltaT)
    Pcab = Pcab / 1000

    n_pack = 2
    min_ventilation = n_pax * 0.55 * lbmin_2_kgs / n_pack

    if m < min_ventilation:
        print('Insuficient mass')
        m = min_ventilation

    gamma = 1.4
    eta_d = 0.84
    eta_c = 0.82
    hx_eff = 0.8
    eta_t = 0.77

    rho, P1, T1 = atmosferaISA(H_m, deltaT)

    T1 = -57 + 273
    P1 = 20  # Kpa

    T2, P2, w_r = ram_compression(m, T1, P1, mach, gamma, eta_d)

    P3 = 250

    T3a, P3a, w_pc = compressor(m, T2, P2, P3 / P2, gamma, eta_c)

    T3b = 200 + 247  # Fixed
    P3b = P3a  # No loss in preecooler

    P4a = 200  # Pressure after PRSOV
    T4a = T3b  # Same temperature after PRSOV

    P4b = P4a  # No pressure drop in hx
    T4b, Qphx = heat_exchanger(m, T4a, T2, hx_eff)

    P7 = Pcab

    P50 = np.array([100])
    resfunc = lambda P5: get_residual(m, T2, P4b, T4b, P5, P7, gamma, hx_eff, eta_c, eta_t, alpha)

    sol = solver(resfunc, np.array([P50]))
    P5 = sol.x[0]
    P6 = P5
    P7 = Pcab

    T5, T6, T7, Qshx, w_t, w_sc = loop(m, T2, P4b, T4b, P5, P7, gamma, hx_eff, eta_c, eta_t)

    m_recirc = e_recirc * m
    m_mixer = m_recirc + m

    T8 = (m_recirc * Cp * Tcab * 1.1 + m * Cp * T7) / (m_mixer * Cp)
    P8 = P7
    Qcooling = m_mixer * Cp * (Tcab - T8)

    # Pressurization work
    _, _, w_press = compressor(m, T2, P2, P7 / P2, gamma, eta_c)
    w_press = w_press + w_r
    COPP = Qcooling / (w_r + w_pc)
    COP = Qcooling / (w_r + w_pc - w_press)

    T = np.array([T1, T2, T3a, T3b, T4a, T4b, T5, T6, T7, T8, Tcab])
    P = np.array([P1, P2, P3a, P3b, P4a, P4b, P5, P6, P7, P8, Pcab])
    W = np.array([w_r, w_pc, w_sc, w_t, w_press])
    s = np.zeros_like(T)

    for idx, _ in enumerate(T):
        s[idx] = PropsSI('S', 'T', T[idx], 'P', P[idx] * 1000, 'air') / 1000

    label = ('1', '2', '3a', '3b', '4a', '4b', '5', '6', '7', '8', 'cab')

    if print_results:
        for idx in range(0, len(P)):
            print('T%s: %.2f P%s: %.2f' % (label[idx], T[idx], label[idx], P[idx]))

        print('\nTurbine Work: %.2f kW\nCompressor Work: %.2f kW' % (w_t, w_sc))
        print('Cooling effect: %.2f kW\t %.2f BTU/hr' % (Qcooling, Qcooling * kW2btumin))
        print('COP %.2f \tCOPP %.2f' % (COP, COPP))

    return W, T, P, s, Qcooling, COP, COPP


def objective_cooling(mach, m, e_recirc, H_cab_ft, Tcab, n_pax, Qrequired, alpha, print_results=False):
    W, T, P, s, Qcooling, COP, COPP = simple_bootstrap_cycle(mach, m, e_recirc, H_cab_ft, Tcab, n_pax, Qrequired, alpha,
                                                             print_results=False)

    return abs(Qcooling - Qrequired)


def calculating_cooling():
    e_recirc = 1
    n_pax = 200

    Qrequired = 100
    alpha = 1
    m = 0.45
    H_cab_ft = 8000
    Tcab = 24 + 273
    mach = 0.78
    min_ventilation = n_pax * 0.55 * lbmin_2_kgs / 2

    W, T, P, s, Qcooling, COP, COPP = simple_bootstrap_cycle(mach, m, e_recirc, H_cab_ft, Tcab, n_pax, Qrequired, alpha,
                                                             print_results=False)

    resfun = lambda m: objective_cooling(mach, m, e_recirc, H_cab_ft, Tcab, n_pax, Qrequired, alpha,
                                         print_results=False)

    bnds = [[min_ventilation, np.inf]]
    sol = minimize(resfun, x0=np.array([.5]), method='trust-constr', bounds=bnds)
    print(sol)
    m = sol.x[0]

    print('Mass required cooling:', m)


def plot_cycle(s, T):
    # End calculation

    # Plotting T-S
    label = ('1', '2', '3a', '3b', '4a', '4b', '5', '6', '7', '8', 'cab')

    _, ax = plt.subplots()
    ax.plot(s, T, 'b-s', linewidth=.8)

    for idx, (s_, T_) in enumerate(zip(s, T)):
        ax.annotate('   ' + label[idx], (s_, T_))

    ax.set_xlabel('s')
    ax.set_ylabel('T')

    plot = PropertyPlot('Air', 'TS', unit_system='KSI', tp_limits='ACHP', axis=ax)
    plot.calc_isolines(CoolProp.iP, iso_range=s, num=(len(s) - 4))

    ax.set_xlim([np.min(s) * 0.97, np.max(s) * 1.03])
    ax.set_ylim([np.min(T) * 0.97, np.max(T) * 1.03])

    plot.show()


def simple_test_plot():
    e_recirc = 0.5
    n_pax = 200

    Qrequired = 61
    alpha = 1
    m = 0.45
    H_cab_ft = 8000
    Tcab = 24 + 273
    mach = 0.78

    W, T, P, s, Qcooling, COP, COPP = simple_bootstrap_cycle(mach, m, e_recirc, H_cab_ft, Tcab, n_pax, Qrequired, alpha,
                                                             print_results=False)

    print(COP, COPP, Qcooling)


def vary_mach():
    alpha = 1
    m = 1.2
    H_cab_ft = 8000
    Tcab = 24 + 273

    mach_range = np.linspace(0.47, 1.8, 20)
    COPP_vec = np.zeros_like(mach_range)
    COP_vec = np.zeros_like(mach_range)
    for idx, mach in enumerate(mach_range):
        W, T, P, s, Qcooling, COP, COPP = simple_bootstrap_cycle(mach, m, H_cab_ft, Tcab, alpha)
        COP_vec[idx] = COP
        COPP_vec[idx] = COPP

    fig, ax = plt.subplots()

    ax.plot(mach_range, COP_vec, 'k-o', mach_range, COPP_vec, 'b-o')
    ax.grid(True)
    ax.set_xlabel('Mach')
    ax.set_ylabel('COP')
    ax.legend(['COP', 'COPP'])
    plt.show()


if __name__ == "__main__":
    #simple_test_plot()
    calculating_cooling()
    # vary_mach()

    # plot_cycle(s,T)
