from scipy import sparse
import numpy as np

#left side is Eikonal
#right side is Brownian

def CreateInterfase(om, k, N, I):
  for i in range(1, N-1):
    om[i, I(i)] = k

def Build_Eik_Brown(om, N):
  for i in range (1, N-1):
    j = 1
    while om[i,j] != 2 and j<N-1: #Eikonal
      om[i,j] = 1
      j = j+1
    j = j+1
    while j < N-1:  #Brownian
      om[i,j] = 3
      j = j+1

#Functions to convert the Browninan domain into a vector and its system

def HashTablesBrownian(om, N):
  #Create hash tables for Brownian
  C_to_P = {}
  P_to_C = {}
  dim = 0
  for i in range(N):
    for j in range(N):
      if om[i,j] == 3:
        C_to_P[i,j] = dim
        P_to_C[dim] = [i,j]
        dim = dim + 1
  return C_to_P, P_to_C, dim      

#Give coordinates and receive position in vector
def Coord_to_Pos(x, y, C_to_P):
  return C_to_P[x,y]

#Give position in vector and receive coordinates
def Pos_to_Coord(l, P_to_C):
  return P_to_C[l]

#Create sparse matrix for Laplace equation in the Brownian region
# the matrix is in CSR format
def CreateMatrixBrownian(C_to_P, P_to_C, dim, om):
  #Create sparse matrix for Brownian
  A = sparse.csr_matrix((dim, dim))
  A = A.tolil()

  #Construct linear system for the Brownian region
  for l in range(dim):
    A[l,l] = -4
    x, y = Pos_to_Coord(l, P_to_C)
    #Check if (x,y) has a neighbor to the left and if it's in the Brownian region
    if om[x, y-1] == 3:
      A[l, Coord_to_Pos(x, y-1, C_to_P)] = 1
    #Check if (x,y) has a neighbor to the right and if it's in the Brownian region
    if om[x, y+1] == 3:
      A[l, Coord_to_Pos(x, y+1, C_to_P)] = 1
    #Check if (x,y) has a neighbor upwards and if it's in the Brownian region
    if om[x-1, y] == 3:
      m = Coord_to_Pos(x-1, y, C_to_P)
      A[l, m] = 1
    #Check if (x,y) has a neighbor downwards and if it's in the Brownian region
    if om[x+1, y] == 3:
      m = Coord_to_Pos(x+1, y, C_to_P)
      A[l, m] = 1

  A = A.tocsr()
  return A

#Functions to update Interfase
def Update_b(sol, CtoP, dim, om, I, N):
  b = np.zeros(dim)
  for i in range(1, N-1):
    x = i
    y = I(i)
    l = Coord_to_Pos(x, y+1, CtoP)
    b[l] = b[l] - sol[x, y]
    if om[x+1, y+1] != 3:
      b[l] = b[l] - sol[x+1, y+1]
    if om[x-1, y+1] != 3:
      b[l] = b[l] - sol[x-1, y+1]
  return b

#Receives vector obtained when solving Laplaces equation and copies it to the solution matrix
# Update_sol_I only updates near interfase
def Update_sol_I(sol, sol_brow, dim, N, CtoP, I, om):
  #print(CtoP)
  for i in range(1, N-1):
    x = i
    y = I(i)
    #print(x, y)
    l = Coord_to_Pos(x, y+1, CtoP)
    sol[x, y+1] = sol_brow[l]
    if om[x+1, y] == 3:
      #print(x+1, y)
      l = Coord_to_Pos(x+1, y, CtoP)
      sol[x+1, y] = sol_brow[l]
    if om[x-1, y] == 3:
      #print(x-1, y+1)
      l = Coord_to_Pos(x-1, y, CtoP)
      sol[x-1, y] = sol_brow[l]
  return sol

def Update_sol(sol, sol_brow, dim, PtoC):
  for l in range(dim):
    x, y = Pos_to_Coord(l, PtoC)
    sol[x, y] = sol_brow[l]
  return sol
