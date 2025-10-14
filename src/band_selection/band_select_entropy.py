# Started by Usman Zahidi (25/02/2025) uzahidi@lincoln.ac.uk

import numpy as np
from joblib import Parallel, delayed

def FuzzyC_MI(image, lambda_param, N):
    Bsize = image.shape[2]
    m = 2

    Img = image
    S_cluster = Bsize // N  # Size of cluster
    redun = Bsize % (N * S_cluster)

    I_ij = MI4FuzzyCImage(image)  # function for Entropy&MI
    I = I_ij
    X_MI = np.sum(I, axis=1) - np.diag(I)
    Mi = np.array_split(X_MI, N-1) + [X_MI[-(redun):]] if redun > 0 else np.array_split(X_MI, N)

    Ind_central = []
    for p in range(len(Mi)):
        max_mi = np.argmax(Mi[p])
        Ind_central.append(max_mi + p * S_cluster)  # S_cluster for location

    Boundry = np.arange(0, S_cluster * N + 1, S_cluster)

    for i in range(100):
        U = Rel(I, Ind_central, m)

        K = RestMI(I, Boundry)
        Ind_NR = np.zeros(len(K), dtype=int)

        for t in range(len(K)):
            V = U[Boundry[t]+1:Boundry[t+1], t] * K[t]
            Ind_new = np.argmax(V)
            Ind_NR[t] = Ind_new + Boundry[t]

        New_Boundry = np.zeros(len(Boundry) - 2)

        for k in range(len(Ind_NR) - 1):
            if (Ind_NR[k+1] - Ind_NR[k] != 1):
                Abso_E = np.vstack((np.abs(I[Ind_NR[k]+1:Ind_NR[k+1]-1, Ind_NR[k]] - I[Ind_NR[k]+1:Ind_NR[k+1]-1, Ind_NR[k+1]]), 
                                      np.arange(Ind_NR[k]+1, Ind_NR[k+1])))
                a, b = np.min(Abso_E, axis=1), np.argmin(Abso_E, axis=1)
                New_Boundry[k] = Abso_E[1, b[0]]
            else:
                New_Boundry[k] = Ind_NR[k]

        Boundry = np.array([0, *New_Boundry, Bsize])
        if np.array_equal(Ind_central, Ind_NR):
            break
        else:
            Ind_central = Ind_NR

def MI4FuzzyCImage(image):
    bandsize = image.shape[2]
    img = image / np.max(image)
    img = img * 255
    Image_u8 = img.astype(np.uint8)
    I_ij = np.zeros((bandsize, bandsize))
    
    def compute_joint_entropy(i, j):
        I11 = Image_u8[:, :, i]
        I22 = Image_u8[:, :, j]
        # Assuming joint_entropyCF is defined elsewhere
        _, joint_entropy, _, _ = joint_entropyCF(I11, I22)
        return joint_entropy

    for i in range(bandsize):
        I_ij[i, :] = Parallel(n_jobs=-1)(delayed(compute_joint_entropy)(i, j) for j in range(bandsize))
    
    return I_ij


