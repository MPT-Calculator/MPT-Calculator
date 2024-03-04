# James Elgy - 02/05/2023

import numpy as np
from matplotlib import pyplot as plt
import netgen.meshing as ngmeshing
from ngsolve import *
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def eval_along_line(E1Mag, E2Mag, E3Mag, mesh, Omega, sigma):
    # line = np.linspace(-3.436189889907837, 3.436189889907837, 1001)
    line = np.linspace(-0.99, 0.99, 1001)
    E1Mag_scaled = E1Mag * Omega * sigma
    E2Mag_scaled = E2Mag * Omega * sigma
    E3Mag_scaled = E3Mag * Omega * sigma
    line_interp_E1 = []
    line_interp_E2 = []
    line_interp_E3 = []

    for point in line:
        mip = mesh(point, 0, 0)
        val1 = E1Mag_scaled(mip)
        val2 = E2Mag_scaled(mip)
        val3 = E3Mag_scaled(mip)
        line_interp_E1 += [val1]
        line_interp_E2 += [val2]
        line_interp_E3 += [val3]
    line_interp_E1 = np.asarray(line_interp_E1)
    line_interp_E2 = np.asarray(line_interp_E2)
    line_interp_E3 = np.asarray(line_interp_E3)

    plt.figure()
    plt.plot(line, line_interp_E1, label='$i=1$')
    plt.plot(line, line_interp_E2, label='$i=2$')
    plt.plot(line, line_interp_E3, label='$i=3$')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel(r'$|\boldsymbol{\theta}^{(1)}_i|$')


if __name__ == '__main__':
    pass
