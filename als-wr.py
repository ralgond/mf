import numpy as np
import pandas as pd
import random
import movielens_data

# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - np.dot(P.T,Q)))**2)/len(R[R > 0]))

class AlsMF():
    def __init__(self, R, T, rank=20, reg=0.1, n_iter = 15):      
        # training and test matrix
        self.R = md.R
        self.T = md.T
        self.m, self.n = R.shape # Number of users and items
        self.rank = rank # Dimensionality of latent feature space
        self.reg = 0.1 # Regularisation weight
        self.n_iter = n_iter # Number of iteration

        # Index matrix for training data
        I = R.copy()
        I[I > 0] = 1
        I[I == 0] = 0

        self.I = I


        # Index matrix for test data
        I2 = T.copy()
        I2[I2 > 0] = 1
        I2[I2 == 0] = 0

        self.I2 = I2

    def train(self):
        P = 1 * np.random.rand(self.rank, self.m) # Latent user feature matrix
        Q = 1 * np.random.rand(self.rank, self.n) # Latent movie feature matrix
        Q[0,:] = self.R[self.R != 0].mean(axis=0) # Avg. rating for each movie
        E = np.eye(self.rank)

        train_errors = []
        test_errors = []

        # Repeat until convergence
        for iter in range(self.n_iter):
            # Fix Q and estimate P
            for i, Ii in enumerate(self.I):
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
                R_Ii = self.R[i, Ii_nonzero]
                Ai = np.dot(Q_Ii, Q_Ii.T) + self.reg * nui * E
                Vi = np.dot(Q_Ii, R_Ii.T)
                #-------------------------------------------------------------------
                
                P[:, i] = np.linalg.solve(Ai, Vi)
            
            # Fix P and estimate Q
            for j, Ij in enumerate(self.I.T):
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
                R_Ij = self.R[Ij_nonzero, j]
                Aj = np.dot(P_Ij, P_Ij.T) + self.reg * nmj * E
                Vj = np.dot(P_Ij, R_Ij)
                #-----------------------------------------------------------------------
                
                Q[:,j] = np.linalg.solve(Aj,Vj)
            
            train_rmse = rmse(self.I, self.R, Q, P)
            test_rmse = rmse(self.I2, self.T, Q, P)
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)
            
            print("[Epoch %d/%d] train error: %f, test error: %f"%(iter+1, self.n_iter, train_rmse, test_rmse))
            
        print("Algorithm converged")

if __name__ == "__main__":
    md = movielens_data.MovielensData('data/movie_rating.csv')
    mf = AlsMF(md.R, md.T)
    mf.train()