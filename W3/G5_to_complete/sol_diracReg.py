import math
import numpy as np

def G5_sol_diracReg(x, epsilon ):
    # Dirac function of x
    # sol_diracReg(x, epsilon) Computes the derivative of the heaviside
    # function of x with respect to x.Regularized based on epsilon.

    # THIS IS GIVEN ON THE SLIDE 43
    y = (1/np.pi) * (epsilon / (epsilon**2 + x**2)) # TO DO 19: Line to complete

    return y