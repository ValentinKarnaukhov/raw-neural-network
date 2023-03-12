from sklearn.preprocessing import MinMaxScaler


class PlainScaler(MinMaxScaler):

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X