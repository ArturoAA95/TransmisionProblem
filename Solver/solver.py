import numpy as np
from DomainConstruction.domain import Update_sol_I, Update_b, CreateInterfase, Update_sol
from FastMarching.fast_marching import FastMarchingMethod2D, InitializeDistances
from scipy.sparse.linalg import spsolve

def Update_Interfase(sol_0, N, I, H):
  for i in range(1, N-1):
    x = i
    y = I(i)
    a_1 = sol_0[x, y+1]
    a_2 = sol_0[x, y-1]
    a_3 = sol_0[x+1, y]
    a_4 = sol_0[x-1, y]
    aux1 = H(a_1, a_2, a_3, a_4, x, y)
    if sol_0[x, y] > aux1:
      sol_0[x, y] = aux1

def Update_Interfase(sol_0, N, I, H):
  for i in range(1, N-1):
    x = i
    y = I(i)
    a_1 = sol_0[x, y+1]
    a_2 = sol_0[x, y-1]
    a_3 = sol_0[x+1, y]
    a_4 = sol_0[x-1, y]
    aux1 = H(a_1, a_2, a_3, a_4, x, y)
    if sol_0[x, y] > aux1:
      sol_0[x, y] = aux1

def InitializeOnePlayer(omega, CtoP, PtoC, N, H, dim, I, A):
  O = np.copy(omega)   ### We preserve the structure of the domain 
  #Solve Eikonal equation in Brownian region
  dist = InitializeDistances(O, N)
  NB = []
  sol_0 = FastMarchingMethod2D(NB, O, dist, N, H) 
  #Solve Laplace equation in Brownian region
  b = Update_b(sol_0, CtoP, dim, O, I, N)
  sol_brow = spsolve(A, b)
  sol_0 = Update_sol(sol_0, sol_brow, dim, PtoC)

  return sol_0

def Solve_One_Player(omega, iter, dim, N, A, CtoP, PtoC, I, H):
  #Initialize solution
  sol_0 = InitializeOnePlayer(omega, CtoP, PtoC, N, H, dim, I, A)
  #Fix the interphase
  CreateInterfase(omega, 0, N, I)
  #Iterate over iter
  for k in range(iter):
    print(k)
    #Update interfase
    Update_Interfase(sol_0, N, I, H)
    #Save information of the domain
    om = np.copy(omega)
    #Solve in the Eikonal region
    sol_0 = FastMarchingMethod2D([], om, sol_0, N, H)    
    #Update the boundary for the Brownian region
    b = Update_b(sol_0, CtoP, dim, om, I, N)
    #Solve Laplace equation in Brownian region
    sol_brow = spsolve(A, b)
    #Update solution
    sol_0 = Update_sol_I(sol_0, sol_brow, dim, N, CtoP, I, om)
    
  sol_0 = Update_sol(sol_0, sol_brow, dim, PtoC)
    
  return sol_0