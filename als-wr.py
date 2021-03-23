import numpy as np
import pandas as pd
import random
import movielens_data

md = movielens_data.MovielensData('data/movie_rating.csv')

# training and test matrix
R = md.R
T = md.T


# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0


# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0


# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))


lmbda = 0.1 # Regularisation weight
k = 20 # Dimensionality of latent feature space
m, n = R.shape # Number of users and items
n_epochs = 15 # Number of epochs

P = 1 * np.random.rand(k,m) # Latent user feature matrix
Q = 1 * np.random.rand(k,n) # Latent movie feature matrix
Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie
E = np.eye(k) # (k x k)-dimensional idendity matrix


train_errors = []
test_errors = []

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix Q and estimate P
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
        if (nui == 0): nui = 1 # Be aware of zero counts!
    
        # Least squares solution
        
        # Replaced lines
        #-----------------------------------------------------------
        # Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E
        # Vi = np.dot(Q, np.dot(np.diag(Ii), R[i].T))
        #-----------------------------------------------------------
        
        # Added Lines
        #-------------------------------------------------------------------
        # Get array of nonzero indices in row Ii
        Ii_nonzero = np.nonzero(Ii)[0]
        # Select subset of Q associated with movies reviewed by user i
        Q_Ii = Q[:, Ii_nonzero]
        # Select subset of row R_i associated with movies reviewed by user i
        R_Ii = R[i, Ii_nonzero]
        Ai = np.dot(Q_Ii, Q_Ii.T) + lmbda * nui * E
        Vi = np.dot(Q_Ii, R_Ii.T)
        #-------------------------------------------------------------------
        
        P[:, i] = np.linalg.solve(Ai, Vi)
    
    # Fix P and estimate Q
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        if (nmj == 0): nmj = 1 # Be aware of zero counts!
        
        # Least squares solution
        
        # Removed Lines
        #-----------------------------------------------------------
        # Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
        # Vj = np.dot(P, np.dot(np.diag(Ij), R[:,j]))
        #-----------------------------------------------------------
        
        # Added Lines
        #-----------------------------------------------------------------------
        # Get array of nonzero indices in row Ij
        Ij_nonzero = np.nonzero(Ij)[0]
        # Select subset of P associated with users who reviewed movie j
        P_Ij = P[:, Ij_nonzero]
        # Select subset of column R_j associated with users who reviewed movie j
        R_Ij = R[Ij_nonzero, j]
        Aj = np.dot(P_Ij, P_Ij.T) + lmbda * nmj * E
        Vj = np.dot(P_Ij, R_Ij)
        #-----------------------------------------------------------------------
        
        Q[:,j] = np.linalg.solve(Aj,Vj)
    
    train_rmse = rmse(I,R,Q,P)
    test_rmse = rmse(I2,T,Q,P)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    
    print("[Epoch %d/%d] train error: %f, test error: %f"%(epoch+1, n_epochs, train_rmse, test_rmse))
    
print("Algorithm converged")