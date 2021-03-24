import numpy as np
import movielens_data

def RMSE(R, R2):
    t_m = ((R-R2)**2)
    sum = 0.
    not_zero_cnt = 0
    rows = R.shape[0]
    cols = R.shape[1]
    for r in range(rows):
        for c in range(cols):
            if (R[r][c] != 0.):
                sum += t_m[r][c]
                not_zero_cnt += 1
    return np.sqrt(sum/not_zero_cnt)



# Calculate the RMSE
def rmse(I,R,U,M):
    return np.sqrt(np.sum((I * (R - np.dot(U,M.T)))**2)/len(R[R > 0]))

class SdgMF():
    def __init__(self, R, T, rank=20, reg=0.1, n_iter=10, learning_rate=0.001):
        self.R = R
        self.T = T
        self.n_users, self.n_items = R.shape
        self.rank = rank
        self.reg = reg
        self.n_iter = n_iter
        self.learning_rate = learning_rate

        self.sample_row, self.sample_col = self.R.nonzero()

        # Index matrix for training data
        I = self.R.copy()
        I[I > 0] = 1
        I[I == 0] = 0

        self.I = I

        # Index matrix for test data
        I2 = self.T.copy()
        I2[I2 > 0] = 1
        I2[I2 == 0] = 0

        self.I2 = I2

    def train(self):
        self.user_vecs = np.random.rand(self.n_users, self.rank)
        self.item_vecs = np.random.rand(self.n_items, self.rank)

        for iter in range(self.n_iter):
            self.sdg()

            train_rmse = rmse(self.I, self.R, self.user_vecs, self.item_vecs)
            test_rmse = rmse(self.I2, self.T, self.user_vecs, self.item_vecs)
            print("tran_rsme:", train_rmse, "test_rmse:", test_rmse)
    
    def sdg(self):
        for idx in range(len(self.sample_row)):
            u, i = self.sample_row[idx], self.sample_col[idx]
            e = self.R[u, i] - self.user_vecs[u,:].dot(self.item_vecs[i,:].T)

            self.user_vecs[u, :] += self.learning_rate * \
                    (e*self.item_vecs[i,:]- self.reg*self.user_vecs[u,:]) 
            self.item_vecs[i, :] += self.learning_rate * \
                    (e*self.user_vecs[u,:]- self.reg*self.item_vecs[i,:])


if __name__ == "__main__":
    md = movielens_data.MovielensData('data/movie_rating.csv')
    mf = SdgMF(md.R, md.T, n_iter=20, learning_rate=0.001)
    mf.train()