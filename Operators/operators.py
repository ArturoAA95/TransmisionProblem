from scipy.optimize import root_scalar
from scipy.optimize import brentq
import numpy as np


def H(a1, a2, a3, a4):
  return min(a1, a2, a3, a4) + 1

def S(a1, a2, a3, a4):
  return .25*(a1+a2+a3+a4)

def G(z, a1, a2, a3, a4):
  return np.sqrt(max( 0, z-a1, z-a2)**2 + max(0, z-a3, z-a4)**2 ) - 1

def FindZero(a1, a2, a3, a4, G):
  # Find the root of G(z) = 0
  sol = brentq(G, args=(a1, a2, a3, a4), a=min(a1, a2, a2, a4), b = min(a1, a2, a2, a4)+ 1)
  return sol
