import numpy as np
from scipy.special import jv
import sympy as sym


def exact_sphere(alpha, epsilon, mur, sigma, omega, framework='hyperbolic'):
    """
    Function to calculate the mpt for a sphere of radius alpha at radial frequency omega.
    Addapted from a Matlab function from Paul Ledger (exactsphererev5freqscan_nod.m).
    :param omega - Angular frequency rad/s
    :param epsilon - Permittivity:
    :param sigma - Conductivity S/m:
    :param mur - Relative Permeability:
    :param alpha - Sphere radius m:
    :param framework='hyperbolic' method of calculation ('bessel', 'negative', 'symbolic', 'hyperbolic') used when 
    calculating eigenvalue. Use symbolic to avoid infs arising from large k (slightly slower).
    :return eig - Single unique eigenvalue of the mpt tensor for a sphere of radius alpha:
    """

    mu0 = 4 * np.pi * (1e-7)
    mu = mur * mu0
    k = np.sqrt((omega ** 2 * epsilon * sigma) + (mu * sigma * omega) * 1j)

    if framework == 'bessel':
        js_0_kr = np.sqrt(np.pi / (2 * k * alpha)) * jv(1 / 2,
                                                             k * alpha)  # jv is the bessel function of the first kind.
        js_1_kr = np.sqrt(np.pi / (2 * k * alpha)) * jv(3 / 2, k * alpha)
        js_2_kr = np.sqrt(np.pi / (2 * k * alpha)) * jv(5 / 2, k * alpha)
        mpt = (2 * np.pi * alpha ** 3) * (2 * (mu - mu0) * js_0_kr + (2 * mu + mu0) * js_2_kr) / (
                (mu + 2 * mu0) * js_0_kr + (mu - mu0) * js_2_kr)
        return mpt
    elif framework == 'negative':
        k *= alpha
        numerator = (2 * mur + 1) * (np.sinh(k) - k * np.cosh(k)) + k ** 2 * np.sinh(k)
        denominator = (mur - 1) * (np.sinh(k) - k * np.cosh(k)) - k ** 2 * np.sinh(k)
        mpt = -np.pi * 2 * (alpha ** 3) * numerator / denominator
        return mpt
    elif framework == 'symbolic':
        # Using sympy to try to avoid overflow errors.
        k *= alpha
        alpha = alpha

        js_0_kr = sym.sqrt(sym.pi / (2 * k)) * sym.besselj(1 / 2, k)
        js_1_kr = sym.sqrt(sym.pi / (2 * k)) * sym.besselj(3 / 2, k)
        js_2_kr = sym.sqrt(sym.pi / (2 * k)) * sym.besselj(5 / 2, k)
        mpt = (2 * np.pi * alpha ** 3) * (2 * (mu - mu0) * js_0_kr + (2 * mu + mu0) * js_2_kr) / (
                (mu + 2 * mu0) * js_0_kr + (mu - mu0) * js_2_kr)
        return complex(mpt.evalf())


    else:
        Ip12 = np.sqrt(2 / np.pi / (k * alpha)) * np.sinh(k * alpha)
        Im12 = np.sqrt(2 / np.pi / (k * alpha)) * np.cosh(k * alpha)

        # sinh and cosh return inf for large values of k*alpha. This leads to the numerator and denominator both being inf.
        # inf/inf is undefined so eig = nan.
        numerator = ((2 * mu + mu0) * k * alpha * Im12 - (mu0 * (1 + (k * alpha) ** 2) + 2 * mu) * Ip12)
        denominator = ((mu - mu0) * k * alpha * Im12 + (mu0 * (1 + (k * alpha) ** 2) - mu) * Ip12)
        eig = 2 * np.pi * alpha ** 3 * numerator / denominator
        eig = np.conj(eig)

        return eig