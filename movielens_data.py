import pandas as pd

class MovielensData:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        header = ['user_id', 'item_id', 'rating']
        df = pd.read_csv('data/movie_rating.csv', names=header)
        self.data = pd.DataFrame(df)
        self.n_users = df.user_id.max()
        self.n_items = df.item_id.max()
        print ('Number of users = ' + str(self.n_users) + ' | Number of movies = ' + str(self.n_items))
