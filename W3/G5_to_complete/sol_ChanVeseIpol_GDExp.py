import cv2
import numpy as np
import math
import sol_diracReg
import sol_DiFwd
import sol_DiBwd
import sol_DjFwd
import sol_DjBwd
import os
from scipy import ndimage
import matplotlib.pyplot as plt
    

def G5_sol_ChanVeseIpol_GDExp(I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni):
    #Implementation of the Chan - Vese segmentation following the explicit % gradient descent in the paper of Pascal Getreur "Chan-Vese Segmentation".
    #It is the equation 19 from that paper
    # I: Gray color image to segment
    # phi_0: Initial phi
    # mu: mu length parameter(regularizer term)
    # nu: nu area parameter(regularizer term)
    # eta: epsilon for the total variation regularization
    # lambda1, lambda2: data fidelity parameters
    # tol: tolerance for the sopping criterium
    # epHeaviside: epsilon for the regularized heaviside.
    # dt: time step
    # iterMax: MAximum number of iterations
    # reIni: Iterations for reinitialization. 0 means no reinitializacion

    folderInput = './'

    ni = I.shape[0]
    nj = I.shape[1]

    hi = 1
    hj = 1

    phi_f = phi_0.copy()
    dif = math.inf
    nIter = 0
    phi_old = phi_f.copy()

    dif_dict = dict()
    images = []

    # WE WILL KEEP ITERATING THE PROBLEM UNTIL WE REACH THE TOTAL
    # MAXIMUM NUMBER OF ITERATIONS OR UNTIL WE REACH A SATISFACTORY
    # RESULT, WHICH WILL BE WHEN THE DIFFERENCE BETWEEN TWO ITERATIONS
    # IS LOWER THAN A PREVIOUSLY SET TOLERANCE
    while dif > tol and nIter < iterMax:

        nIter = nIter + 1
        print(nIter)
    
        # WE HAD PHI NORMALIZED BETWEEN [-1, 1], SO NOW THE FIRST STEP IS TO
        # SEPARATE THE INTIAL IMAGE IN TWO: THE VALUES THAT ARE LARGER OR EQUAL
        # TO ZERO AND THE VALUES WHICH ARE LOWER THAN ZERO
        I1 = I[phi_f >= 0]
        I0 = I[phi_f < 0]
        I1f = I1.astype('float')
        I0f = I0.astype('float')

        # Minimization w.r.t c1 and c2(constant estimation)

        # WE HAVE SEPARATED TWO REGIONS OF THE ORIGINAL IMAGE (I1F, LARGER OR EQUAL TO ZERO
        # AND I2F, LOWER THAN ZERO), FOR THE CONSTANT VALUE TO REPRESENT EACH REGION WE CAN
        # THE MEAN VALUE OF EACH REGION, OR MAYBE THA MAXIMUM IN THE >= 0 REGION AND THE
        # MINIMUM IN THE <0 REGION. 
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #WHAT THE PAPER SAYS:
        # and ϕ. For fixed ϕ, the optimal values of c1 and c2 are the region av
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        # TODO 1: Line to complete
        # TODO 2: Line to complete

        H = 0.5*(1 + (2/np.pi)*np.arctan(phi_old/epHeaviside))
        c1 = np.sum(I*H)/np.sum(H) 
        c2 = np.sum(I*(1-H))/np.sum(1-H) 

        
        #
        c1 = np.mean(I1f)
        c2 = np.mean(I0f)

        # Boundary conditions

        # WE NEED THE BOUNDARIES TO BE SMOOTH, SO WE NEED EACH
        # BOUNDARY PIXEL TO BE THE SAME (OR VERY SIMILAR) TO THE
        # ONE WHICH IS CLOSEST AND IS FROM THE SAME REGION 
        phi_f[0, :] = phi_f[1, :]        #TODO 3: Line to complete
        phi_f[ni-1, :] = phi_f[ni-2, :]     #TODO 4: Line to complete

        phi_f[:, 0] = phi_f[:, 1]        #TODO 5: Line to complete
        phi_f[:, nj-1] = phi_f[:, nj-2]     #TODO 6: Line to completend)


        
       
        # phi_f variable is mutable along the function. A trick to avoid this:
        phi_c=phi_f.copy()
        
        new_phi2 = np.zeros(phi_c.shape, dtype=float)
        new_phi3 = np.zeros(phi_c.shape, dtype=float)
        new_phi4 = np.zeros(phi_c.shape, dtype=float)
        new_phi = np.zeros(phi_c.shape, dtype=float)
        phi = np.zeros(phi_c.shape, dtype=float)
        for i in range(0, phi_c.shape[0]):
            for j in range(0, phi_c.shape[1]):
                new_phi[i, j] = phi_c[i, j]
                new_phi2[i, j] = phi_c[i, j]
                new_phi3[i, j] = phi_c[i, j]
                new_phi4[i, j] = phi_c[i, j]
                phi[i, j] = phi_c[i, j]
        
        '''
        # OUR CODE MODIFICATION
        # phi_f variable is mutable along the function. A trick to avoid this:
        phi_c=phi_f[:]

        new_phi0 = np.zeros(phi_c.shape, dtype= float)
        new_phi1 = np.zeros(phi_c.shape, dtype= float)
        new_phi2 = np.zeros(phi_c.shape, dtype= float)
        new_phi3 = np.zeros(phi_c.shape, dtype= float)
        new_phi4 = np.zeros(phi_c.shape, dtype= float)
        new_phi1 = np.zeros(phi_c.shape, dtype= float)
        phi = np.zeros(phi_c.shape, dtype= float)
        # WE THINK THAT THESE NEW PHIS SHOULD BE USED TO COMPUTE THE 
        # FORWARD AND BACKWARD GRADIENTS. AND WE HAVE CHANGED THEIR NAMES
        # IN ORDER TO BE MORE EASILY RECOGNIZABLE
        for i in range(0, phi_c.shape[0]):
            for j in range(0, phi_c.shape[1]):
                new_phi0[i, j] = phi_c[i, j]
                new_phi1[i, j] = phi_c[i, j]
                new_phi2[i, j] = phi_c[i, j]
                new_phi3[i, j] = phi_c[i, j]
                phi[i, j] = phi_c[i, j]
                new_phi4[i, j] = phi_c[i, j]
                
        '''

        # Regularized Dirac 's Delta computation
        delta_phi = sol_diracReg.G5_sol_diracReg(phi_c, epHeaviside) # H'(phi)

        # THE FORWARD DIFFERENCE FORMULA IS THE NEXT PIXEL MINUS
        # THE CURRENT PIXEL, AND THE BACKWARD FORMULA IS THE CURRENT
        # PIXEL MINUS THE EARLIER PIXEL
        # derivatives estimation
        # i direction
        phi_iFwd = sol_DiFwd.G5_sol_DiFwd(new_phi, hi)   #TODO 7: Line to complete
        phi_iBwd = sol_DiBwd.G5_sol_DiBwd(new_phi4, hi)   #TODO 8: Line to complete

        # j direction
        phi_jFwd = sol_DjFwd.G5_sol_DjFwd(new_phi2, hj)   #TODO 9: Line to complete
        phi_jBwd = sol_DjBwd.G5_sol_DjBwd(new_phi3, hj)   #TODO 10: Line to complete

        # THE CENTERED DIFFERENCE IS THE SUM OF THE FORWARD AND THE
        # BACKWWARD DIFFERENCE DIVIDED BY 2
        # centered
        phi_icent = (phi_iFwd + phi_iBwd)/2.0  #TODO 11: Line to complete
        phi_jcent = (phi_jFwd + phi_jBwd)/2.0  #TODO 12: Line to complete

        # ETA IS 10**-8 IN THE PAPER BUT WE ARE CURRENTLY USING 0.01
        #A and B estimation (A y B from the Pascal Getreuer's IPOL paper "Chan Vese segmentation
        A = mu / np.sqrt(eta**2 + phi_jFwd**2 + phi_icent**2)      #TODO 13: Line to complete
        B = mu / np.sqrt(eta**2 + phi_jcent**2 + phi_iFwd**2)      #TODO 14: Line to complete

        #Equation 22, for inner points
        # SLIDE 46, WE NEED TO COMPUTE THE NEW PHI FOR THE INNER POINTS
        # SINCE EARLIER WE SET THE VALUES FOR THE BOUNDARIES

        # CREO QUE LAS A Y B ESTAS BIEN PERO LAS PHIS NO 

        enumerator = (phi_old[1:ni-1, 1:nj-1] +
                     dt*delta_phi[1:ni-1, 1:nj-1]*(
                     A[1:ni-1, 1:nj-1]*phi_old[2:ni,1:nj-1] +
                     A[0:ni-2, 1:nj-1]*phi[0:ni-2, 1:nj-1] + 
                     B[1:ni-1, 1:nj-1]*phi_old[1:ni-1,2:nj] + 
                     B[1:ni-1, 0:nj-2]*phi[1:ni-1,0:nj-2] - 
                     nu -
                     lambda1*(I[1:ni-1, 1:nj-1] - c1)**2 +
                     lambda2*(I[1:ni-1, 1:nj-1] - c2)**2))


        denominator = 1 + dt*delta_phi[1:ni-1, 1:nj-1]*(
                        A[1:ni-1, 1:nj-1] +
                        A[0:ni-2, 1:nj-1] + 
                        B[1:ni-1, 1:nj-1] + 
                        B[1:ni-1, 0:nj-2]) 

        phi[1:ni-1, 1:nj-1] = enumerator / denominator

        if reIni > 0 & np.mod(nIter, reIni) == 0:

            indGT1 = phi >= 0
            indGT = indGT1.astype('float')
            indLT1 = phi< 0
            indLT=indLT1.astype('float')

            xb1 = ndimage.distance_transform_edt(1-indLT)
            xb2 = ndimage.distance_transform_edt(1-indGT)

            phi = (xb1-xb2)

            # Normalization[-1,1]
            nor = min(abs(phi.min()), phi.max())

            # Normalize `phi` by dividing it by `nor`
            phi = phi / nor

            #Diference. This stopping criterium has the problem that phi can
            #change, but not the zero level set, that it really is what we are looking for

            dif = np.mean(np.sum((phi.ravel() - phi_old.ravel())**2))
            phi_old = phi.copy()
            phi_f=phi.copy()
            seg = phi>=0
            seg = seg.astype('float')
            print(dif)
            dif_dict[nIter] = dif
            images.append((seg * 255).astype("int"))
            #cv2.imshow('Seg',seg)
            #cv2.waitKey(0)
        else:
            phi_old=phi.copy()

        

    seg = phi>=0
    seg = seg.astype('uint8')

    return seg, dif_dict, images




