from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import sys
import cv2
import math

@dataclass
class Parameters:
    hi: float
    hj: float
    dt: float
    iterMax: float
    tol: float

def sol_Laplace_Equation_Axb_MATLAB(f, dom2Inp, param):

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

    # cv2.imshow('dom2Inp', dom2Inp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p)
        a_ij.insert(idx, 1)
        idx = idx + 1

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p + (ni + 2))
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
        idx_Aj.insert(idx, p - (ni + 2))
        a_ij.insert(idx, -1)
        idx = idx + 1

        b[p-1] = 0

    # Inner points
    for j in range(2, nj + 2, 1):
        for i in range(2, ni + 2, 1):

            # from image matrix (i, j) coordinates to vectorial(p) coordinate
            p = (j - 1) * (ni + 2) + i

            if (dom2Inp_ext[i-1, j-1] > 0): # If we have to inpaint this pixel
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5

                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p)
                a_ij.insert(idx, 4)
                idx = idx + 1

                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p + 1)
                a_ij.insert(idx, -1)
                idx = idx + 1

                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p - 1)
                a_ij.insert(idx, -1)
                idx = idx + 1

                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p + (ni + 2))
                a_ij.insert(idx, -1)
                idx = idx + 1

                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p - (ni + 2))
                a_ij.insert(idx, -1)
                idx = idx + 1

                idx_i = i 

                idx_j = j 

                b[p-1] = 4 * dom2Inp_ext[idx_i, idx_j] - (dom2Inp_ext[idx_i-1, idx_j] + dom2Inp_ext[idx_i, idx_j-1] + dom2Inp_ext[idx_i+1, idx_j] + dom2Inp_ext[idx_i, idx_j+1])
                #print("{}: {}".format(p-1, b[p-1]))
                
                
            else:
                # we do not have to inpaint this pixel
                # TO COMPLETE 6
                idx_Ai.insert(idx, p)
                idx_Aj.insert(idx, p)
                a_ij.insert(idx, 1)
                idx = idx + 1
                # print(f_ext[i-1, j-1])
                # print(f_ext[i-1, j-1].dtype)
                b[p-1] = f_ext[i-1, j-1]

    # to start the array at 0
    idx_Ai_c = [i - 1 for i in idx_Ai]
    idx_Aj_c = [i - 1 for i in idx_Aj]

    # A is a sparse matrix, so  for memory requirements we create a sparse matrix
    # TO COMPLETE 7

    A = sparse(idx_Ai_c, idx_Aj_c, a_ij, (ni+2), (nj+2)) # ??? and ???? is the size of matrix A
    # Solve the sistem of equations
    x = spsolve(A, b)

    # From vector to matrix
    u_ext = np.reshape(x,(ni+2, nj+2),order='F')
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]

    # Eliminate the ghost boundaries
    u = np.full((ni, nj), u_ext[1:u_ext_i-1, 1:u_ext_j-1], order='F')
    return u



def G5_sparse(i, j, v, rows, cols):
    return csr_matrix((v, (i, j)), shape=(rows*cols, rows*cols))





def G5_sol_Laplace_Equation_Axb_python_seamless(f, dom2Inp, binary_mask, param):

    # this code is not intended to be  efficient.

    ni = f.shape[0]
    nj = f.shape[1]

    mi = dom2Inp.shape[0]
    mj = dom2Inp.shape[1]

    #We add the ghost boundaries (for the boundary conditions)

    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]

    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f  # array starts at 0 in Python, in Matlab 1

    dom2Inp_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = dom2Inp_ext.shape[0]
    ndj_ext = dom2Inp_ext.shape[1]
    dom2Inp_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = dom2Inp

    # binary mask
    bin_mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = bin_mask_ext.shape[0]
    ndj_ext = bin_mask_ext.shape[1]
    bin_mask_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = binary_mask

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

            if bin_mask_ext[i, j] > 0:
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5

                # Boundaries of B:       
                if ((bin_mask_ext[i + 1, j] == 0) or
                   (bin_mask_ext[i - 1, j] == 0) or
                   (bin_mask_ext[i, j + 1] == 0) or
                   (bin_mask_ext[i, j - 1] == 0)):
                    
                    idx_Ai.append(p)
                    idx_Aj.append(p)
                    a_ij.append(1)

                    b[p] = f_ext[i, j]


                else:
                        
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
                    
                    grad = 4 * dom2Inp_ext[i, j] - (dom2Inp_ext[i-1, j] + dom2Inp_ext[i, j-1] + dom2Inp_ext[i+1, j] + dom2Inp_ext[i, j+1]) 
                    b[p] = grad
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










def G5_sol_Laplace_Equation_Axb_python_mixed_seamless(f, dom2Inp, binary_mask, param):

    # this code is not intended to be  efficient.

    ni = f.shape[0]
    nj = f.shape[1]

    mi = dom2Inp.shape[0]
    mj = dom2Inp.shape[1]

    #We add the ghost boundaries (for the boundary conditions)

    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]

    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f  # array starts at 0 in Python, in Matlab 1

    dom2Inp_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = dom2Inp_ext.shape[0]
    ndj_ext = dom2Inp_ext.shape[1]
    dom2Inp_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = dom2Inp

    # binary mask
    bin_mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = bin_mask_ext.shape[0]
    ndj_ext = bin_mask_ext.shape[1]
    bin_mask_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = binary_mask

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

            grad = 4 * dom2Inp_ext[i, j] - (dom2Inp_ext[i-1, j] + dom2Inp_ext[i, j-1] + dom2Inp_ext[i+1, j] + dom2Inp_ext[i, j+1]) 

            if bin_mask_ext[i, j] > 0:
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5
                               
                if (bin_mask_ext[i + 1, j] == 0) or (bin_mask_ext[i - 1, j] == 0) or (bin_mask_ext[i, j+ 1] == 0) or (bin_mask_ext[i, j - 1] == 0):
                    
                    if ((bin_mask_ext[i + 1, j] == 0) or
                       (bin_mask_ext[i - 1, j] == 0) or
                       (bin_mask_ext[i, j + 1] == 0) or
                       (bin_mask_ext[i, j - 1] == 0)):
                    
                        idx_Ai.append(p)
                        idx_Aj.append(p)
                        a_ij.append(1)

                        b[p] = f_ext[i, j]

                else:
                        
                    if abs(grad) > 0:
                        
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
                        
                        
                        b[p] = grad

                    else:
                        # we do not have to inpaint this pixel
                        # TO COMPLETE 6
                        idx_Ai.append(p)
                        idx_Aj.append(p)
                        a_ij.append(1)

                        b[p] = f_ext[i, j]


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








def G5_sol_Laplace_Equation_Axb_python_mixing_gradients(f, dom2Inp, binary_mask, param):

    # this code is not intended to be  efficient.

    ni = f.shape[0]
    nj = f.shape[1]

    mi = dom2Inp.shape[0]
    mj = dom2Inp.shape[1]

    #We add the ghost boundaries (for the boundary conditions)

    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]

    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f  # array starts at 0 in Python, in Matlab 1

    dom2Inp_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = dom2Inp_ext.shape[0]
    ndj_ext = dom2Inp_ext.shape[1]
    dom2Inp_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = dom2Inp

    # binary mask
    bin_mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = bin_mask_ext.shape[0]
    ndj_ext = bin_mask_ext.shape[1]
    bin_mask_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = binary_mask

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

            grada = 4 * f_ext[i, j] - (f_ext[i-1, j] + f_ext[i, j-1] + f_ext[i+1, j] + f_ext[i, j+1])
            gradb = 4 * dom2Inp_ext[i, j] - (dom2Inp_ext[i-1, j] + dom2Inp_ext[i, j-1] + dom2Inp_ext[i+1, j] + dom2Inp_ext[i, j+1])

            if bin_mask_ext[i, j] > 0:
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5
                               
                if ((bin_mask_ext[i + 1, j] == 0) or
                   (bin_mask_ext[i - 1, j] == 0) or
                   (bin_mask_ext[i, j + 1] == 0) or
                   (bin_mask_ext[i, j - 1] == 0)):
                    
                    idx_Ai.append(p)
                    idx_Aj.append(p)
                    a_ij.append(1)

                    b[p] = f_ext[i, j]


                else:

                    if abs(grada) == 0 and abs(gradb) == 0:
                        idx_Ai.append(p)
                        idx_Aj.append(p)
                        a_ij.append(1)
                        b[p] = f_ext[i,j]
                    else:

                        if abs(grada) > abs(gradb):
                            idx_Ai.append(p)
                            idx_Aj.append(p)
                            a_ij.append(1)
                            
                            b[p] = f_ext[i, j]
                        else:
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
                            b[p] = gradb
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





def G5_sol_Laplace_Equation_Axb_python_seamless_and_dest_averaged(f, dom2Inp, binary_mask, param):
    # this code is not intended to be  efficient.

    ni = f.shape[0]
    nj = f.shape[1]

    mi = dom2Inp.shape[0]
    mj = dom2Inp.shape[1]

    #We add the ghost boundaries (for the boundary conditions)

    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]

    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f  # array starts at 0 in Python, in Matlab 1

    dom2Inp_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = dom2Inp_ext.shape[0]
    ndj_ext = dom2Inp_ext.shape[1]
    dom2Inp_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = dom2Inp

    # binary mask
    bin_mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = bin_mask_ext.shape[0]
    ndj_ext = bin_mask_ext.shape[1]
    bin_mask_ext[1: ndi_ext - 1, 1: ndj_ext - 1] = binary_mask

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

            if bin_mask_ext[i, j] > 0:
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5
                               
                if (bin_mask_ext[i + 1, j] == 0) or (bin_mask_ext[i - 1, j] == 0) or (bin_mask_ext[i, j+ 1] == 0) or (bin_mask_ext[i, j - 1] == 0):
                    
                    if ((bin_mask_ext[i + 1, j] == 0) or
                       (bin_mask_ext[i - 1, j] == 0) or
                       (bin_mask_ext[i, j + 1] == 0) or
                       (bin_mask_ext[i, j - 1] == 0)):
                    
                        idx_Ai.append(p)
                        idx_Aj.append(p)
                        a_ij.append(1)

                        b[p] = f_ext[i, j]


                else:
                        
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
                    
                    gradb = 4 * dom2Inp_ext[i, j] - (dom2Inp_ext[i-1, j] + dom2Inp_ext[i, j-1] + dom2Inp_ext[i+1, j] + dom2Inp_ext[i, j+1]) 
                    grada = 4 * f_ext[i, j] - (f_ext[i-1, j] + f_ext[i, j-1] + f_ext[i+1, j] + f_ext[i, j+1]) 
                    b[p] = (grada + gradb)/2
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













# INPAINTING DONE CONSIDERING 3 BORDERS, DONE AT THE BEGINNING SINCE WE DID NOT
# KNOW WE HAD TO USE BINARY MASKS TOO. NOT USED CURRENTLY
def G5_sol_Laplace_Equation_Axb_python_3borderd(f, dom2Inp, param):

    # this code is not intended to be  efficient.

    ni = f.shape[0]
    nj = f.shape[1]

    mi = dom2Inp.shape[0]
    mj = dom2Inp.shape[1]

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

            grad = 4 * dom2Inp_ext[i, j] - (dom2Inp_ext[i-1, j] + dom2Inp_ext[i, j-1] + dom2Inp_ext[i+1, j] + dom2Inp_ext[i, j+1]) 
        

            if  abs(grad)>0:
                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and % vector b
                # TO COMPLETE 5
              
                if (dom2Inp_ext[i + 1, j] == 0) or (dom2Inp_ext[i - 1, j] == 0) or (dom2Inp_ext[i, j + 1] == 0) or (dom2Inp_ext[i, j - 1] == 0):
                    # Border Level 1 (Smooth b=a)
                    idx_Ai.append(p)
                    idx_Aj.append(p)
                    a_ij.append(1)
                    b[p] = f_ext[i, j] 

                elif (dom2Inp_ext[i + 2, j] == 0) or (dom2Inp_ext[i - 2, j] == 0) or (dom2Inp_ext[i, j + 2] == 0) or (dom2Inp_ext[i, j - 2] == 0):
                    # Border Level 2 (Smooth b=0)

                    idx_Ai.append(p)
                    idx_Aj.append(p)
                    a_ij.append(4)

                    idx_Ai.append(p)
                    idx_Aj.append(p + 1)
                    a_ij.append(-1)

                    idx_Ai.append(p)
                    idx_Aj.append(p - 1)
                    a_ij.append(-1)
                    idx = idx + 1

                    idx_Ai.append(p)
                    idx_Aj.append(p + (ni + 2))
                    a_ij.append(-1)

                    idx_Ai.append(p)
                    idx_Aj.append(p - (ni + 2))
                    a_ij.append(-1)

                    idx_i = i
                    idx_j = j
                    b[p] = 0
                elif (dom2Inp_ext[i + 3, j] == 0) or (dom2Inp_ext[i - 3, j] == 0) or (dom2Inp_ext[i, j + 3] == 0) or (dom2Inp_ext[i, j - 3] == 0):
                    # Border Level 3 (Smooth b=0)
                    idx_Ai.append(p)
                    idx_Aj.append(p)
                    a_ij.append(4)

                    idx_Ai.append(p)
                    idx_Aj.append(p + 1)
                    a_ij.append(-1)

                    idx_Ai.append(p)
                    idx_Aj.append(p - 1)
                    a_ij.append(-1)

                    idx_Ai.append(p)
                    idx_Aj.append(p + (ni + 2))
                    a_ij.append(-1)

                    idx_Ai.append(p)
                    idx_Aj.append(p - (ni + 2))
                    a_ij.append(-1)

                    idx_i = i
                    idx_j = j
                    b[p] = 0

                else: 
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
                    

                    idx_i = i
                    idx_j = j

                    grad =  4 * dom2Inp_ext[i, j] - (dom2Inp_ext[i-1, j] + dom2Inp_ext[i, j-1] + dom2Inp_ext[i+1, j] + dom2Inp_ext[i, j+1]) 
                    
                    b[p] = grad

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


