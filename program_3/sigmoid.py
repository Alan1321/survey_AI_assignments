#---------------------------------------------------
# File: sigmoid.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner, Micah Mayers
# Course: CS430-01
# Date: 10/15/2023
# Python 3.8.10
#---------------------------------------------------

import math
import numpy as np

def sigmoid(z):
    return 1 / (1 + math.e**(-z))



#---------------------------------------------------
# TEST CASES
#---------------------------------------------------
# Exactly 0.5 when z = 0
# ~1 for large, positive values of z
# ~0 for large, negative values of z
#---------------------------------------------------
# Applies to scalars, vectors, and matrices
#---------------------------------------------------
def test():
    values = [0, 100, -100]

    # Test with scalars
    print('\nSCALAR TEST')
    for z in values:
        print(sigmoid(z))
        print()

    # Test with column vectors
    print('\nVECTOR TEST')
    vector = np.zeros((3, 1))
    for z in values:
        vector.fill(z)
        print(sigmoid(vector))
        print()

    # Test with matrices
    print('\nMATRIX TEST')
    matrix = np.zeros((3, 3))
    for z in values:
        matrix.fill(z)
        print(sigmoid(matrix))
        print()