from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    iterMax: float
    tol: float

def G5_sol_Laplace_Equation_Axb(f, dom2Inp, param):

    # this code is not intended to be  efficient.

    ni = f.shape[0]
    nj = f.shape[1]

    #We add the ghost boundaries (for the boundary conditions)

    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]

    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f  # array starts at 0 in Python, in Matlab 1

    dom2Inp_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = dom2Inp_ext.shape[0]
    ndj_ext = dom2Inp_ext.shape[1]
    dom2Inp_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = dom2Inp

    #Store memory for the A matrix and the b vector
    nPixels = (ni+2)*(nj+2); #Number of pixels

    #We will create A sparse, this is the number of nonzero positions

    #idx_Ai: Vector for the nonZero i index of matrix A
    #idx_Aj: Vector for the nonZero j index of matrix A
    #a_ij: Vector for the value at position ij of matrix A

    b = np.zeros((nPixels,1), dtype=float)

    #Vector counter
    idx=0

    idx_Ai=[]
    idx_Aj=[]
    a_ij=[]

    #North side boundary conditions
    i=1
    for j in range(1, nj+3, 1):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = (j - 1) * (ni + 2) + i

    # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx,p)
        a_ij.insert(idx, 1)
        idx = idx + 1

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p + 1)
        a_ij.insert(idx, -1)
        idx = idx + 1

        b[p-1] = 0

# South side boundary conditions
    i = ni + 2
    for j in range(1,nj + 3,1):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = (j - 1) * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
        # TO COMPLETE 2
        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx,p)
        a_ij.insert(idx, 1)
        idx = idx + 1

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p - 1)
        a_ij.insert(idx, -1)
        idx = idx + 1

        b[p-1] = 0

# West side boundary conditions
    j = 1
    for i in range(1,ni + 3,1):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = (j - 1) * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
        # TO COMPLETE 3
        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx,p)
        a_ij.insert(idx, 1)
        idx = idx + 1

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p + (ni+2))
        a_ij.insert(idx, -1)
        idx = idx + 1

        b[p-1] = 0
    # East side boundary conditions
    j = nj + 2
    for i in range(1,ni + 3,1):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = (j - 1) * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
        # TO COMPLETE 4
        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx,p)
        a_ij.insert(idx, 1)
        idx = idx + 1

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p - (ni+2))
        a_ij.insert(idx, -1)
        idx = idx + 1

        b[p-1] = 0

    # Inner points
    for j in range(2, nj + 2, 1):
        for i in range(2, ni + 2, 1):

            # from image matrix (i, j) coordinates to vectorial(p) coordinate
            p = (j - 1) * (ni + 2) + i

            if (dom2Inp_ext[i-1, j-1] == 1): # If we have to inpaint this pixel
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx,p)
                a_ij.insert(idx, 4)
                idx = idx + 1

                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p - 1)
                a_ij.insert(idx, -1)
                idx = idx + 1

                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p + 1)
                a_ij.insert(idx, -1)
                idx = idx + 1

                #idx_Ai.insert(idx, p + (ni + 2))
                idx_Ai.insert(idx, p )
                idx_Aj.insert(idx, p - (ni+2))
                a_ij.insert(idx, -1)
                idx = idx + 1

                #idx_Ai.insert(idx, p - (ni + 2))
                idx_Ai.insert(idx, p )
                idx_Aj.insert(idx, p + (ni+2))
                a_ij.insert(idx, -1)
                idx = idx + 1

                b[p-1] = 0
            else:
                # we do not have to inpaint this pixel
                # TO COMPLETE 6
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p)
                a_ij.insert(idx, 1)
                idx = idx + 1

                b[p-1] = f_ext[i-1, j-1]

    # to start the array at 0
    idx_Ai_c = [i - 1 for i in idx_Ai]
    idx_Aj_c = [i - 1 for i in idx_Aj]

    # A is a sparse matrix, so  for memory requirements we create a sparse matrix
    # TO COMPLETE 7

    A = G5_sparse(idx_Ai_c, idx_Aj_c, a_ij, ni+2, nj+2) # ??? and ???? is the size of matrix A
    # Solve the sistem of equations
    x = spsolve(A, b)

    # From vector to matrix
    u_ext = np.reshape(x,(ni+2, nj+2),order='F')
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]

    # Eliminate the ghost boundaries
    u = np.full((ni, nj), u_ext[1:u_ext_i-1, 1:u_ext_j-1], order='F')
    return u

def G5_sparse(i, j, v, nrows, ncols):
    #print("n_rows: {} \t n_cols: {} \t n_rows * n_cols: {}".format(nrows, ncols, nrows*ncols))
    #print(max(i))
    #print(max(j))
    return csr_matrix((v, (i, j)), shape=(nrows*ncols, nrows*ncols))



def G5_sol_Laplace_Equation_Axb_python(f, dom2Inp, param):

    # this code is not intended to be  efficient.

    ni = f.shape[0]
    nj = f.shape[1]

    #We add the ghost boundaries (for the boundary conditions)

    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]

    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f  # array starts at 0 in Python, in Matlab 1

    dom2Inp_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = dom2Inp_ext.shape[0]
    ndj_ext = dom2Inp_ext.shape[1]
    dom2Inp_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = dom2Inp

    #Store memory for the A matrix and the b vector
    nPixels = (ni+2)*(nj+2); #Number of pixels

    #We will create A sparse, this is the number of nonzero positions

    #idx_Ai: Vector for the nonZero i index of matrix A
    #idx_Aj: Vector for the nonZero j index of matrix A
    #a_ij: Vector for the value at position ij of matrix A

    b = np.zeros((nPixels,1), dtype=float)

    #Vector counter
    idx=0

    idx_Ai=[]
    idx_Aj=[]
    a_ij=[]

    #North side boundary conditions
    i=0
    for j in range(nj+2):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = j * (ni + 2) + i

    # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b

        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p+1)
        a_ij.append(-1)

        b[p] = 0

# South side boundary conditions
    i = ni + 1
    for j in range(nj+2):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = j * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
        # TO COMPLETE 2
        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p-1)
        a_ij.append(-1)

        b[p] = 0

# West side boundary conditions
    j = 0
    for i in range(ni+2):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = j * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
        # TO COMPLETE 3
        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p + (ni + 2))
        a_ij.append(-1)

        b[p] = 0
    # East side boundary conditions
    j = nj + 1
    for i in range(ni + 2):
    # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = j * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
        # TO COMPLETE 4
        
        idx_Ai.append(p)
        idx_Aj.append(p)
        a_ij.append(1)

        idx_Ai.append(p)
        idx_Aj.append(p - (ni + 2))
        a_ij.append(-1)

        b[p] = 0

    # Inner points
    for j in range(1, nj + 1):
        for i in range(1, ni + 1):

            # from image matrix (i, j) coordinates to vectorial(p) coordinate
            p = j * (ni + 2) + i

            if (dom2Inp_ext[i, j] == 1): # If we have to inpaint this pixel
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5
                idx_Ai.append(p)
                idx_Aj.append(p)
                a_ij.append(4)

                idx_Ai.append(p)
                idx_Aj.append(p+1)
                a_ij.append(-1)

                idx_Ai.append(p)
                idx_Aj.append(p-1)
                a_ij.append(-1)

                idx_Ai.append(p)
                idx_Aj.append(p+ni+2)
                a_ij.append(-1)

                idx_Ai.append(p)
                idx_Aj.append(p-(ni+2))
                a_ij.append(-1)

                b[p] = 0
            else:
                # we do not have to inpaint this pixel
                # TO COMPLETE 6
                idx_Ai.append(p)
                idx_Aj.append(p)
                a_ij.append(1)

                b[p] = f_ext[i, j]

    # to start the array at 0
    idx_Ai_c = [i - 1 for i in idx_Ai]
    idx_Aj_c = [i - 1 for i in idx_Aj]

    # A is a sparse matrix, so  for memory requirements we create a sparse matrix
    # TO COMPLETE 7

    A = G5_sparse(idx_Ai, idx_Aj, a_ij, ni+2, nj+2) # ??? and ???? is the size of matrix A
    # Solve the sistem of equations
    x = spsolve(A, b)

    # From vector to matrix
    u_ext = np.reshape(x,(ni+2, nj+2),order='F')
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]

    # Eliminate the ghost boundaries
    u = np.full((ni, nj), u_ext[1:u_ext_i-1, 1:u_ext_j-1], order='F')
    return u
