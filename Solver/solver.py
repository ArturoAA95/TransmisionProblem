import numpy as np
from DomainConstruction.domain import Update_sol_I, Update_b, CreateInterfase
from FastMarching.fast_marching import FastMarchingMethod2D
from scipy.sparse.linalg import spsolve

def Update_Interfase(sol_0, N, f, H):
  for i in range(1, N-1):
    x = i
    y = f(i)
    a_1 = sol_0[x, y+1]
    a_2 = sol_0[x, y-1]
    a_3 = sol_0[x+1, y]
    a_4 = sol_0[x-1, y]
    aux1 = H(a_1, a_2, a_3, a_4)
    if sol_0[x, y] > aux1:
      sol_0[x, y] = aux1

def InitialSol(omega, N, f, H):
  return pass

def Solve_One_Player(omega, sol_0, iter, dim, N, A, PtoC, f, H):
  #Iterate over iter
  for k in range(iter):
    print(k)
    #Update interphase
    Update_Interfase(sol_0, N, f, H)
    #Save information of the domain
    om = np.copy(omega)
    #Solve in the Eikonal region
    sol_0 = FastMarchingMethod2D([], om, sol_0, N, H)    
    #Update the boundary for the Brownian region
    b = Update_b(sol_0, PtoC, dim, om, f, N)
    #Solve Laplace equation in Brownian region
    sol_brow = spsolve(A, b)
    #Update solution
    sol_0 = Update_sol_I(sol_0, sol_brow, dim, N, PtoC, f)
    
  return sol_0