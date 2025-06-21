import numpy as np
import heapq
from Operators.operators import FindZero, H, G

#NB: Narrow band
#om: A rectangular domain omega with labels 1,2,3 and 4.
  # 1 and 2 correspond to the eikonal part
#N: Length of the domain
#dist: Initialized matrix of distances to the boundary
#H: Operator to compute the distance to the boundary

def InitializeNarrowBand(NB, om, dist, N, H):
  for i in range(N):
    for j in range(N):
      if om[i, j] > 0 and om[i, j] < 3: #Run over eikonal
        if om[i+1, j] == 0:
            aux = H(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1])
        elif om[i-1, j] == 0:
            aux = H(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1])
        elif om[i, j+1] == 0:
            aux = H(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1])
        elif om[i, j-1] == 0:
            aux = H(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1])
        else:
          continue  #The index is not added to the narrow band
        new_dist = np.copy(aux)
        if new_dist < dist[i, j]:
          dist[i, j] = new_dist
          x = [i, j]
          heapq.heappush(NB, (new_dist, x))

def UpdateDistance(NB, om, dist, N, H, i_aux, j_aux):
    #Update the distance of i_aux, j_aux
    new_dist = H(dist[i_aux+1, j_aux], dist[i_aux-1, j_aux], dist[i_aux, j_aux+1], dist[i_aux, j_aux-1])
    #Only update if the distance is smaller than the current distance
    if new_dist < dist[i_aux, j_aux]:
        dist[i_aux, j_aux] = new_dist
        heapq.heappush(NB, (new_dist, [i_aux, j_aux]))


def FastMarchingMethod2D(NB, om, dist, N, H): #Solves Eikonal equation
  #Inicialice narrow band
  InitializeNarrowBand(NB, om, dist, N, H)
  #Compute solution
  #Loop until narrow band is empty
  while len(NB)>0 :
    #Pop the element with the least distance in the narrow band
    c_dist, c_vert = heapq.heappop(NB)
    i = int(c_vert[0])
    j = int(c_vert[1])
    #The vertex (i,j) becomes a boundary point
    om[i, j] = 0
    # Check if we have already poped the vertex
    if c_dist > dist[i, j]:
      continue
    #Update distance for neighbors of current_vertex
    #Update the distance of the neighbor upwards
    if om[i+1, j] > 0 and om[i+1, j] < 3: #Restrict to Eikonal
      i_aux = i+1
      j_aux = j
      UpdateDistance(NB, om, dist, N, H, i_aux, j_aux)
    #Update the distance of the neighbor downwards
    if om[i-1, j] > 0 and om[i-1, j] < 3: #Restrict to Eikonal
      i_aux = i-1
      j_aux = j
      UpdateDistance(NB, om, dist, N, H, i_aux, j_aux)
    #Update the distance of the neighbor to the right
    if om[i, j+1] > 0 and om[i, j+1] < 3: #Restrict to Eikonal
      i_aux = i
      j_aux = j+1
      UpdateDistance(NB, om, dist, N, H, i_aux, j_aux)
    #Update the distance of the neighbor to the left
    if om[i, j-1] > 0 and om[i, j-1] < 3: #Restrict to Eikonal
      i_aux = i
      j_aux = j-1
      UpdateDistance(NB, om, dist, N, H, i_aux, j_aux)
  #Return the distance matrix
  return dist

def InitializeNarrowBandG(NB, om, dist, N, H):
  for i in range(N):
    for j in range(N):
      if om[i, j] > 0 and om[i, j] < 3: #Run over eikonal
        if om[i+1, j] == 0:
            aux = FindZero(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1], H)
        elif om[i-1, j] == 0:
            aux = FindZero(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1], H)
        elif om[i, j+1] == 0:
            aux = FindZero(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1], H)
        elif om[i, j-1] == 0:
            aux = FindZero(dist[i+1, j], dist[i-1, j], dist[i, j+1], dist[i, j-1], H)
        else:
          continue  #The index is not added to the narrow band
        new_dist = np.copy(aux)
        if new_dist < dist[i, j]:
          dist[i, j] = new_dist
          x = [i, j]
          heapq.heappush(NB, (new_dist, x))

def UpdateDistanceG(NB, om, dist, N, H, i_aux, j_aux):
    #Update the distance of i_aux, j_aux
    new_dist = FindZero(dist[i_aux+1, j_aux], dist[i_aux-1, j_aux], dist[i_aux, j_aux+1], dist[i_aux, j_aux-1], H)
    #Only update if the distance is smaller than the current distance
    if new_dist < dist[i_aux, j_aux]:
        dist[i_aux, j_aux] = new_dist
        heapq.heappush(NB, (new_dist, [i_aux, j_aux]))


def FastMarchingMethod2DG(NB, om, dist, N, H): #Solves Eikonal equation
  #Inicialice narrow band
  InitializeNarrowBandG(NB, om, dist, N, H)
  #Compute solution
  #Loop until narrow band is empty
  while len(NB)>0 :
    #Pop the element with the least distance in the narrow band
    c_dist, c_vert = heapq.heappop(NB)
    i = int(c_vert[0])
    j = int(c_vert[1])
    #The vertex (i,j) becomes a boundary point
    om[i, j] = 0
    # Check if we have already poped the vertex
    if c_dist > dist[i, j]:
      continue
    #Update distance for neighbors of current_vertex
    #Update the distance of the neighbor upwards
    if om[i+1, j] > 0 and om[i+1, j] < 3: #Restrict to Eikonal
      i_aux = i+1
      j_aux = j
      UpdateDistanceG(NB, om, dist, N, H, i_aux, j_aux)
    #Update the distance of the neighbor downwards
    if om[i-1, j] > 0 and om[i-1, j] < 3: #Restrict to Eikonal
      i_aux = i-1
      j_aux = j
      UpdateDistanceG(NB, om, dist, N, H, i_aux, j_aux)
    #Update the distance of the neighbor to the right
    if om[i, j+1] > 0 and om[i, j+1] < 3: #Restrict to Eikonal
      i_aux = i
      j_aux = j+1
      UpdateDistanceG(NB, om, dist, N, H, i_aux, j_aux)
    #Update the distance of the neighbor to the left
    if om[i, j-1] > 0 and om[i, j-1] < 3: #Restrict to Eikonal
      i_aux = i
      j_aux = j-1
      UpdateDistanceG(NB, om, dist, N, H, i_aux, j_aux)
  #Return the distance matrix
  return dist

def InitializeDistances(om, N):
   dist = np.zeros((N,N))
   for i in range(N):
    for j in range(N):
       if om[i, j] > 0:
          dist[i, j] = np.inf
   return dist